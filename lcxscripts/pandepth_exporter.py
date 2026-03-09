import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import yaml

# Ensure the examples directory (which contains the `datasets` package) is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
import sys

if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from examples.datasets.colmap import Parser  # type: ignore
from gsplat.rendering import rasterization  # type: ignore


def load_yaml_config(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data)}")
    return data


def load_splats_from_ckpt(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load Gaussian parameters from a gsplat simple_trainer checkpoint.

    This mirrors how Runner.rasterize_splats() prepares inputs:
    - scales are stored in log space and exponentiated before rasterization
    - opacities are stored in logit space and passed through sigmoid
    - colors are SH coefficients (sh0 + shN)
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "splats" not in ckpt:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain 'splats' state_dict; "
            "make sure this is produced by examples.simple_trainer."
        )
    splats_sd = ckpt["splats"]

    means = splats_sd["means"].to(device)  # [N, 3]
    scales_param = splats_sd["scales"].to(device)  # log scale [N, 3]
    quats = splats_sd["quats"].to(device)  # [N, 4]
    opacities_param = splats_sd["opacities"].to(device)  # logit [N]

    if "sh0" in splats_sd and "shN" in splats_sd:
        sh0 = splats_sd["sh0"].to(device)  # [N, 1, 3]
        shN = splats_sd["shN"].to(device)  # [N, K-1, 3]
        colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]
    elif "colors" in splats_sd:
        # Fallback for appearance-optimized checkpoints that may store direct colors
        colors = splats_sd["colors"].to(device)  # [N, 3] or [N, K, 3]
    else:
        raise KeyError(
            "Checkpoint 'splats' must contain either ('sh0', 'shN') or 'colors'."
        )

    scales = torch.exp(scales_param)  # [N, 3]
    opacities = torch.sigmoid(opacities_param)  # [N,] - rasterization expects 1D tensor

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
    }


def make_cube_camera_matrices(
    center_c2w: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Create 6 camera-to-world matrices for a camera-oriented cubemap.

    center_c2w: [4, 4] camera-to-world matrix.
                The cubemap center is at the camera position,
                and the posz face (front) is aligned with the camera's forward direction.
    Return:
        c2w_all: [6, 4, 4]
        names:  list of face names in the order of c2w_all
    """
    # Extract camera position and rotation
    t = center_c2w[:3, 3]  # [3] translation
    R_camera = center_c2w[:3, :3]  # [3, 3] rotation (camera-to-world)

    # Define cubemap face orientations in camera-local coordinates.
    # Each R_local is formed as [right, up, forward] in columns (OpenGL-style).
    # These are relative to the camera's coordinate system.
    # Note: We swap left/right and top/bottom definitions to match equirectangular mapping.
    faces_local = {
        "posz": np.array(  # Front: same as camera forward
            [
                [1, 0, 0],  # right
                [0, 1, 0],  # up
                [0, 0, 1],  # forward
            ],
            dtype=np.float32,
        ),
        "negz": np.array(  # Back: opposite to camera forward
            [
                [-1, 0, 0],  # right
                [0, 1, 0],   # up
                [0, 0, -1],  # forward
            ],
            dtype=np.float32,
        ),
        # Swapped: posx uses left definition, negx uses right definition
        "posx": np.array(  # Right (swapped with left): camera's left direction
            [
                [0, 0, 1],   # right (swapped)
                [0, 1, 0],   # up
                [-1, 0, 0],  # forward (swapped)
            ],
            dtype=np.float32,
        ),
        "negx": np.array(  # Left (swapped with right): camera's right direction
            [
                [0, 0, -1],  # right (swapped)
                [0, 1, 0],   # up
                [1, 0, 0],   # forward (swapped)
            ],
            dtype=np.float32,
        ),
        # Swapped: posy uses bottom definition, negy uses top definition
        "posy": np.array(  # Top (swapped with bottom): camera's down direction
            [
                [1, 0, 0],   # right
                [0, 0, -1],  # up (swapped)
                [0, -1, 0],  # forward (swapped)
            ],
            dtype=np.float32,
        ),
        "negy": np.array(  # Bottom (swapped with top): camera's up direction
            [
                [1, 0, 0],   # right
                [0, 0, 1],   # up (swapped)
                [0, 1, 0],   # forward (swapped)
            ],
            dtype=np.float32,
        ),
    }

    names = list(faces_local.keys())
    mats: List[np.ndarray] = []
    for n in names:
        R_local = faces_local[n]  # [3, 3] local rotation
        # Transform local rotation to world coordinates
        R_world = R_camera @ R_local  # [3, 3]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_world
        c2w[:3, 3] = t
        mats.append(c2w)
    return np.stack(mats, axis=0), names


def make_cube_intrinsics(face_size: int) -> torch.Tensor:
    """Create a pinhole intrinsic matrix for a 90-degree FOV cube face."""
    f = float(face_size) / 2.0
    cx = cy = float(face_size) / 2.0
    K = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K


def render_cubemap_for_view(
    splats: Dict[str, torch.Tensor],
    c2w_center: np.ndarray,
    face_size: int,
    near_plane: float,
    far_plane: float,
    device: torch.device,
    sh_degree: int = 3,
    render_depth: bool = False,
    depth_type: str = "expected",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Render 6 cubemap faces (RGB and optionally depth) around a given center camera pose.
    
    Args:
        depth_type: "expected" for ED (Expected Depth) or "accumulated" for D (Accumulated Depth)
    
    Returns:
        face_images: Dict[str, np.ndarray] - RGB images for each face
        face_depths: Dict[str, np.ndarray] - Depth images for each face (if render_depth=True)
    """
    c2w_faces_np, face_names = make_cube_camera_matrices(c2w_center)
    c2w_faces = torch.from_numpy(c2w_faces_np).to(device=device, dtype=torch.float32)
    viewmats = torch.linalg.inv(c2w_faces)  # [6, 4, 4] world-to-camera

    K_face = make_cube_intrinsics(face_size).to(device)
    Ks = K_face.unsqueeze(0).repeat(len(face_names), 1, 1)  # [6, 3, 3]

    # Debug: Print Gaussian distribution info
    if render_depth:
        means = splats["means"]  # [N, 3]
        cam_center_world = c2w_center[:3, 3]
        # Compute distances from camera center to all Gaussians
        gaussian_dists = torch.norm(means - torch.tensor(cam_center_world, device=device, dtype=means.dtype), dim=1)
        gaussian_dists_np = gaussian_dists.detach().cpu().numpy()
        print(f"    [Pre-render Debug] Gaussian distribution:")
        print(f"      Total Gaussians: {len(means):,}")
        print(f"      Distance from camera center: min={gaussian_dists_np.min():.3f}, max={gaussian_dists_np.max():.3f}, mean={gaussian_dists_np.mean():.3f}")
        print(f"      Distance percentiles: 1%={np.percentile(gaussian_dists_np, 1):.3f}, 50%={np.percentile(gaussian_dists_np, 50):.3f}, 99%={np.percentile(gaussian_dists_np, 99):.3f}")
        print(f"      Gaussians within near_plane: {np.sum(gaussian_dists_np < near_plane):,}")
        print(f"      Gaussians beyond far_plane: {np.sum(gaussian_dists_np > far_plane):,}")
        if far_plane < 1e9:
            print(f"      Gaussians in valid range [{near_plane:.3f}, {far_plane:.3f}]: {np.sum((gaussian_dists_np >= near_plane) & (gaussian_dists_np <= far_plane)):,}")

    # Choose render mode based on depth_type
    if render_depth:
        if depth_type == "accumulated":
            render_mode = "RGB+D"  # Accumulated Depth (D)
        else:
            render_mode = "RGB+ED"  # Expected Depth (ED, default)
    else:
        render_mode = "RGB"
    colors, _, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=splats["scales"],
        opacities=splats["opacities"],
        colors=splats["colors"],
        viewmats=viewmats,
        Ks=Ks,
        width=face_size,
        height=face_size,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        with_ut=False,
        with_eval3d=False,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode=render_mode,
        sh_degree=sh_degree,  # Required when colors are SH coefficients
    )
    # Clamp RGB channels to [0,1] but preserve depth values (they are in world-space units)
    if render_depth:
        # Separate RGB and depth channels
        rgb_channels = colors[..., :3]  # [6, H, W, 3]
        depth_channel = colors[..., 3:4]  # [6, H, W, 1]
        # Only clamp RGB channels
        rgb_channels = torch.clamp(rgb_channels, 0.0, 1.0)
        # Concatenate back: RGB + depth (depth is NOT clamped)
        colors = torch.cat([rgb_channels, depth_channel], dim=-1)  # [6, H, W, 4]
        
        # Debug: Check depth values before any clamping
        all_depths = depth_channel.detach().cpu().numpy().squeeze(-1)  # [6, H, W]
        all_depths_valid = all_depths[all_depths > 0]
        if len(all_depths_valid) > 0:
            depth_min_all = all_depths_valid.min()
            depth_max_all = all_depths_valid.max()
            print(f"    [Post-render Debug] Depth value range across all faces (before clamping):")
            print(f"      Global min: {depth_min_all:.6f}, max: {depth_max_all:.6f}")
            if depth_max_all <= 1.0 and depth_min_all >= 0.0:
                print(f"      [DIAGNOSIS] ⚠️  Depth values are in [0,1] range - may be normalized by rasterization!")
                print(f"         This could explain why some faces show depth ≈ 1.0")
                print(f"         Note: Depth values are now preserved (not clamped) for proper visualization")
            elif depth_max_all > 1.0:
                print(f"      Depth values exceed 1.0, using world-space units (preserved, not clamped)")
    else:
        # No depth channel, just clamp RGB
        colors = torch.clamp(colors, 0.0, 1.0)  # [6, H, W, 3]

    face_images: Dict[str, np.ndarray] = {}
    face_depths: Dict[str, np.ndarray] = {}
    
    for i, name in enumerate(face_names):
        if render_depth:
            # Extract RGB and depth
            rgb = colors[i, ..., :3]  # [H, W, 3]
            depth = colors[i, ..., 3:4]  # [H, W, 1]
            img = (rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)
            depth_np = depth.detach().cpu().numpy().squeeze(-1)  # [H, W]
            
            # Get camera pose for this face
            c2w_face = c2w_faces_np[i]
            cam_pos = c2w_face[:3, 3]
            cam_forward = c2w_face[:3, 2]  # Forward direction in world space
            
            # Debug: Print raw depth statistics immediately after extraction
            valid_mask = depth_np > 0
            total_pixels = depth_np.size
            valid_pixels = np.sum(valid_mask)
            invalid_pixels = total_pixels - valid_pixels
            
            print(f"    [Render Debug] {name} - Raw depth from rasterization:")
            print(f"      Shape: {depth_np.shape}, dtype: {depth_np.dtype}")
            print(f"      Total pixels: {total_pixels}, Valid: {valid_pixels} ({100*valid_pixels/total_pixels:.1f}%), Invalid: {invalid_pixels} ({100*invalid_pixels/total_pixels:.1f}%)")
            print(f"      Camera position: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
            print(f"      Camera forward: [{cam_forward[0]:.3f}, {cam_forward[1]:.3f}, {cam_forward[2]:.3f}]")
            print(f"      Render params: near={near_plane:.6f}, far={far_plane:.2e}")
            
            # Check RGB image statistics to see if there's geometry
            rgb_mean = img.mean(axis=(0, 1))
            rgb_std = img.std(axis=(0, 1))
            rgb_nonzero = np.sum(np.any(img > 0, axis=2))
            rgb_nonblack = np.sum(np.any(img > 10, axis=2))  # Pixels with some visible content
            print(f"      RGB stats: mean=[{rgb_mean[0]:.1f}, {rgb_mean[1]:.1f}, {rgb_mean[2]:.1f}], std=[{rgb_std[0]:.1f}, {rgb_std[1]:.1f}, {rgb_std[2]:.1f}]")
            print(f"      RGB non-zero pixels: {rgb_nonzero}/{total_pixels} ({100*rgb_nonzero/total_pixels:.1f}%)")
            print(f"      RGB non-black pixels (>10): {rgb_nonblack}/{total_pixels} ({100*rgb_nonblack/total_pixels:.1f}%)")
            
            # Cross-check: Compare RGB content with depth validity
            if valid_pixels > 0:
                rgb_valid_mask = np.any(img > 10, axis=2)  # Pixels with visible RGB content
                depth_valid_mask = valid_mask
                both_valid = np.sum(rgb_valid_mask & depth_valid_mask)
                rgb_only = np.sum(rgb_valid_mask & ~depth_valid_mask)
                depth_only = np.sum(~rgb_valid_mask & depth_valid_mask)
                neither = np.sum(~rgb_valid_mask & ~depth_valid_mask)
                print(f"      Pixel correlation: RGB+Depth={both_valid}, RGB-only={rgb_only}, Depth-only={depth_only}, Neither={neither}")
                if depth_only > total_pixels * 0.5:
                    print(f"      [DIAGNOSIS] ⚠️  Many pixels have depth but no RGB content - possible background/empty space")
                if rgb_only > total_pixels * 0.5:
                    print(f"      [DIAGNOSIS] ⚠️  Many pixels have RGB but no depth - possible rendering issue")
            
            if valid_pixels > 0:
                valid_depth = depth_np[valid_mask]
                p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(valid_depth, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                print(f"      Valid depth stats: min={valid_depth.min():.6f}, max={valid_depth.max():.6f}, mean={valid_depth.mean():.6f}, std={valid_depth.std():.6f}")
                print(f"      Percentiles: 1%={p1:.6f}, 5%={p5:.6f}, 10%={p10:.6f}, 25%={p25:.6f}, 50%={p50:.6f}, 75%={p75:.6f}, 90%={p90:.6f}, 95%={p95:.6f}, 99%={p99:.6f}")
                
                # Check for specific suspicious patterns
                depth_range = valid_depth.max() - valid_depth.min()
                
                # Check if all values are exactly the same (or very close)
                unique_values = np.unique(valid_depth)
                print(f"      Unique depth values: {len(unique_values)} (range: [{unique_values.min():.6f}, {unique_values.max():.6f}])")
                
                # Check if depth is close to 1.0 (possible normalization issue)
                close_to_one = np.sum(np.abs(valid_depth - 1.0) < 1e-3)
                if close_to_one > 0:
                    print(f"      [DIAGNOSIS] {close_to_one}/{valid_pixels} ({100*close_to_one/valid_pixels:.1f}%) pixels have depth ≈ 1.0")
                    if close_to_one == valid_pixels:
                        print(f"      [DIAGNOSIS] ⚠️  ALL valid pixels have depth ≈ 1.0! Possible causes:")
                        print(f"         - Depth values may be normalized to [0,1] range")
                        print(f"         - All geometry is at far_plane distance")
                        print(f"         - No geometry in this direction (empty space)")
                
                # Check if depth is close to far_plane
                if far_plane < 1e9:  # Only check if far_plane is reasonable
                    close_to_far = np.sum(np.abs(valid_depth - far_plane) < far_plane * 1e-3)
                    if close_to_far > 0:
                        print(f"      [DIAGNOSIS] {close_to_far}/{valid_pixels} ({100*close_to_far/valid_pixels:.1f}%) pixels have depth ≈ far_plane ({far_plane:.2e})")
                        if close_to_far == valid_pixels:
                            print(f"      [DIAGNOSIS] ⚠️  ALL valid pixels hit far_plane! Geometry may be too far or missing.")
                
                # Check if depth is close to near_plane
                close_to_near = np.sum(np.abs(valid_depth - near_plane) < near_plane * 1e-2)
                if close_to_near > 0:
                    print(f"      [DIAGNOSIS] {close_to_near}/{valid_pixels} ({100*close_to_near/valid_pixels:.1f}%) pixels have depth ≈ near_plane ({near_plane:.6f})")
                
                # Check depth distribution
                if depth_range > 0:
                    # Check if depth is concentrated in a small range
                    depth_span_90 = np.percentile(valid_depth, 95) - np.percentile(valid_depth, 5)
                    depth_span_50 = np.percentile(valid_depth, 75) - np.percentile(valid_depth, 25)
                    print(f"      Depth span: 90% range={depth_span_90:.6f}, 50% range (IQR)={depth_span_50:.6f}")
                    if depth_span_90 / depth_range < 0.1:
                        print(f"      [DIAGNOSIS] ⚠️  Depth values are highly concentrated (90% span is {100*depth_span_90/depth_range:.1f}% of total range)")
                
                # Check for suspicious values
                if valid_depth.max() > 1e6:
                    print(f"      [WARNING] Very large depth values detected (max={valid_depth.max():.2e})!")
                if valid_depth.min() < 1e-6:
                    print(f"      [WARNING] Very small depth values detected (min={valid_depth.min():.2e})!")
                
                if depth_range < 1e-3:
                    print(f"      [WARNING] Depth range is very small ({depth_range:.6f})! Visualization may appear uniform.")
                    print(f"      [DIAGNOSIS] Possible reasons:")
                    print(f"         - All geometry is at the same distance from camera")
                    print(f"         - Depth values are normalized/clamped")
                    print(f"         - No geometry variation in this direction")
            else:
                print(f"      [WARNING] No valid depth values in {name}!")
                print(f"      [DIAGNOSIS] All pixels have depth ≤ 0. Possible causes:")
                print(f"         - No geometry visible in this direction")
                print(f"         - All geometry is behind the camera")
                print(f"         - Rendering issue")
            
            face_images[name] = img
            face_depths[name] = depth_np
        else:
            img = (colors[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
            face_images[name] = img
    
    if render_depth:
        # Print summary across all faces
        print(f"    [Render Debug] Summary across all {len(face_depths)} faces:")
        all_valid_depths = []
        for name, depth_face in face_depths.items():
            valid = depth_face[depth_face > 0]
            if len(valid) > 0:
                all_valid_depths.extend(valid.tolist())
        if len(all_valid_depths) > 0:
            all_valid_depths = np.array(all_valid_depths)
            print(f"      Global depth stats (all faces combined): min={all_valid_depths.min():.6f}, max={all_valid_depths.max():.6f}, mean={all_valid_depths.mean():.6f}, std={all_valid_depths.std():.6f}")
        return face_images, face_depths
    else:
        return face_images, {}


def cubemap_to_equirect(
    cube_faces: Dict[str, np.ndarray],
    pano_h: int,
    pano_w: int,
    cube_depths: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert axis-aligned cubemap faces to an equirectangular panorama.

    cube_faces: dict with keys ['posx', 'negx', 'posy', 'negy', 'posz', 'negz'],
                each value is [Hc, Wc, 3] uint8 for RGB.
    cube_depths: optional dict with same keys, each value is [Hc, Wc] float32 for depth.
    
    Returns:
        pano: [H, W, 3] uint8 RGB panorama
        pano_depth: [H, W] float32 depth panorama (if cube_depths is provided)
    """
    # Pre-fetch face size and convert to float for indexing.
    any_face = next(iter(cube_faces.values()))
    face_h, face_w, _ = any_face.shape

    # Output panorama
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    pano_depth: Optional[np.ndarray] = None
    if cube_depths is not None:
        pano_depth = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        # Debug: Print input cubemap depth statistics before conversion
        print(f"    [Cubemap->Equirect Debug] Input cubemap depth statistics:")
        for face_name, face_depth in cube_depths.items():
            valid = face_depth[face_depth > 0]
            if len(valid) > 0:
                print(f"      {face_name}: min={valid.min():.6f}, max={valid.max():.6f}, mean={valid.mean():.6f}, valid_pixels={len(valid)}/{face_depth.size}")
            else:
                print(f"      {face_name}: No valid depth values")

    # Create grids of longitude (theta) and latitude (phi).
    # theta: [-pi, pi], phi: [-pi/2, pi/2]
    ys, xs = np.meshgrid(
        np.arange(pano_h, dtype=np.float32),
        np.arange(pano_w, dtype=np.float32),
        indexing="ij",
    )
    theta = (xs / pano_w - 0.5) * 2.0 * math.pi
    phi = (0.5 - ys / pano_h) * math.pi

    # Direction vectors in world coordinates.
    dx = np.cos(phi) * np.sin(theta)
    dy = np.sin(phi)
    dz = np.cos(phi) * np.cos(theta)

    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    abs_dz = np.abs(dz)

    # Determine which face each direction hits.
    is_x_major = (abs_dx >= abs_dy) & (abs_dx >= abs_dz)
    is_y_major = (abs_dy > abs_dx) & (abs_dy >= abs_dz)
    is_z_major = (abs_dz > abs_dx) & (abs_dz > abs_dy)

    # Initialize indices for each panorama pixel into cube faces.
    u = np.zeros_like(dx)
    v = np.zeros_like(dy)
    face_index = np.empty(dx.shape, dtype=np.int32)

    # +X / -X
    # For +X, direction ~ (+1, dy, dz)
    # For -X, direction ~ (-1, dy, dz)
    posx_mask = is_x_major & (dx > 0)
    negx_mask = is_x_major & (dx < 0)
    # +Y / -Y
    posy_mask = is_y_major & (dy > 0)
    negy_mask = is_y_major & (dy < 0)
    # +Z / -Z
    posz_mask = is_z_major & (dz > 0)
    negz_mask = is_z_major & (dz < 0)

    # Compute (u, v) in [-1, 1] for each face (OpenGL convention).
    # Note: Our cubemap faces are camera-oriented, where posz is the front (camera forward).
    # Reference: depth-anything-3's cubemap_to_equirect implementation
    # +X (right face): right=+Z, up=+Y, forward=+X
    if np.any(posx_mask):
        u[posx_mask] = -dz[posx_mask] / abs_dx[posx_mask]  # Fixed: flip sign to match standard convention
        v[posx_mask] = -dy[posx_mask] / abs_dx[posx_mask]
        face_index[posx_mask] = 0  # posx
    # -X (left face): right=-Z, up=+Y, forward=-X
    if np.any(negx_mask):
        u[negx_mask] = dz[negx_mask] / abs_dx[negx_mask]  # Fixed: flip sign to match standard convention
        v[negx_mask] = -dy[negx_mask] / abs_dx[negx_mask]
        face_index[negx_mask] = 1  # negx
    # +Y (top face): right=+X, up=+Z, forward=+Y
    if np.any(posy_mask):
        u[posy_mask] = dx[posy_mask] / abs_dy[posy_mask]
        v[posy_mask] = dz[posy_mask] / abs_dy[posy_mask]  # Fixed: flip sign to match standard convention
        face_index[posy_mask] = 2  # posy
    # -Y (bottom face): right=+X, up=-Z, forward=-Y
    if np.any(negy_mask):
        u[negy_mask] = dx[negy_mask] / abs_dy[negy_mask]
        v[negy_mask] = -dz[negy_mask] / abs_dy[negy_mask]  # Fixed: flip sign to match standard convention
        face_index[negy_mask] = 3  # negy
    # +Z (front face): right=+X, up=+Y, forward=+Z
    if np.any(posz_mask):
        u[posz_mask] = dx[posz_mask] / abs_dz[posz_mask]
        v[posz_mask] = -dy[posz_mask] / abs_dz[posz_mask]
        face_index[posz_mask] = 4  # posz
    # -Z (back face): right=-X, up=+Y, forward=-Z
    if np.any(negz_mask):
        u[negz_mask] = -dx[negz_mask] / abs_dz[negz_mask]
        v[negz_mask] = -dy[negz_mask] / abs_dz[negz_mask]
        face_index[negz_mask] = 5  # negz

    # Convert (u, v) in [-1, 1] to pixel coordinates in [0, face_w-1] / [0, face_h-1].
    uu = ((u + 1.0) * 0.5) * (face_w - 1)
    vv = ((v + 1.0) * 0.5) * (face_h - 1)

    # Prepare list of faces in the same order as indices above.
    ordered_faces = [
        cube_faces["posx"],
        cube_faces["negx"],
        cube_faces["posy"],
        cube_faces["negy"],
        cube_faces["posz"],
        cube_faces["negz"],
    ]
    ordered_depths = None
    if cube_depths is not None:
        ordered_depths = [
            cube_depths["posx"],
            cube_depths["negx"],
            cube_depths["posy"],
            cube_depths["negy"],
            cube_depths["posz"],
            cube_depths["negz"],
        ]

    # Sample by nearest neighbor for simplicity.
    uu_i = np.clip(np.round(uu).astype(np.int32), 0, face_w - 1)
    vv_i = np.clip(np.round(vv).astype(np.int32), 0, face_h - 1)

    # Direct mapping: no face index swapping needed since we swapped definitions in make_cube_camera_matrices
    # But we may need to flip UV coordinates for top/bottom faces to align edges correctly
    for fi, face_img in enumerate(ordered_faces):
        mask = face_index == fi
        if not np.any(mask):
            continue
        
        # For top/bottom faces, flip vertically to align edges with side faces
        if fi == 2:  # posy (top)
            # Flip vertically (up-down) to align edges
            vv_flipped = face_h - 1 - vv_i[mask]
            pano[mask] = face_img[vv_flipped, uu_i[mask]]
            if pano_depth is not None and ordered_depths is not None:
                pano_depth[mask] = ordered_depths[fi][vv_flipped, uu_i[mask]]
        elif fi == 3:  # negy (bottom)
            # Flip vertically (up-down) to align edges
            vv_flipped = face_h - 1 - vv_i[mask]
            pano[mask] = face_img[vv_flipped, uu_i[mask]]
            if pano_depth is not None and ordered_depths is not None:
                pano_depth[mask] = ordered_depths[fi][vv_flipped, uu_i[mask]]
        else:  # posx, negx, posz, negz
            pano[mask] = face_img[vv_i[mask], uu_i[mask]]
            if pano_depth is not None and ordered_depths is not None:
                pano_depth[mask] = ordered_depths[fi][vv_i[mask], uu_i[mask]]

    # Debug: Print output panorama depth statistics after conversion
    if pano_depth is not None:
        valid_mask = pano_depth > 0
        total_pano_pixels = pano_depth.size
        valid_pano_pixels = np.sum(valid_mask)
        print(f"    [Cubemap->Equirect Debug] Output panorama depth statistics:")
        print(f"      Shape: {pano_depth.shape}, dtype: {pano_depth.dtype}")
        print(f"      Total pixels: {total_pano_pixels}, Valid: {valid_pano_pixels} ({100*valid_pano_pixels/total_pano_pixels:.1f}%)")
        if valid_pano_pixels > 0:
            valid_pano_depth = pano_depth[valid_mask]
            p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(valid_pano_depth, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            print(f"      Valid depth stats: min={valid_pano_depth.min():.6f}, max={valid_pano_depth.max():.6f}, mean={valid_pano_depth.mean():.6f}, std={valid_pano_depth.std():.6f}")
            print(f"      Percentiles: 1%={p1:.6f}, 5%={p5:.6f}, 10%={p10:.6f}, 25%={p25:.6f}, 50%={p50:.6f}, 75%={p75:.6f}, 90%={p90:.6f}, 95%={p95:.6f}, 99%={p99:.6f}")
        else:
            print(f"      [WARNING] No valid depth values in output panorama!")

    return pano, pano_depth


def visualize_depth_map_gsplat_style(
    depth: np.ndarray,
    color_space: str = "linear",
    debug_label: str = "",
) -> np.ndarray:
    """Convert depth map to grayscale visualization (gsplat video style).
    
    This matches the depth visualization used in gsplat's render_traj:
    - Normalize depth to [0, 1] using min/max
    - Convert to grayscale (repeat 3 channels)
    
    Args:
        depth: [H, W] float32 depth map
        color_space: "linear" for linear depth, "log" for log depth visualization
        debug_label: Optional label for debug output
    
    Returns:
        [H, W, 3] uint8 grayscale depth visualization
    """
    # Create valid mask (depth > 0)
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # No valid depth, return black image
        h, w = depth.shape
        if debug_label:
            print(f"    [Visualize Debug] {debug_label} (gsplat_style): No valid depth values!")
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Debug: Print input depth statistics
    valid_depth = depth[valid_mask]
    if debug_label:
        print(f"    [Visualize Debug] {debug_label} (gsplat_style) - Input depth:")
        print(f"      Valid pixels: {np.sum(valid_mask)}/{depth.size} ({100*np.sum(valid_mask)/depth.size:.1f}%)")
        print(f"      Raw depth: min={valid_depth.min():.6f}, max={valid_depth.max():.6f}, mean={valid_depth.mean():.6f}, std={valid_depth.std():.6f}")
    
    # Apply color space transformation
    depth_vis = depth.copy()
    if color_space == "log":
        # Log depth space (similar to depth-anything-3 style)
        depth_vis[valid_mask] = np.log(depth_vis[valid_mask])
        if debug_label:
            log_depth = depth_vis[valid_mask]
            print(f"      After log transform: min={log_depth.min():.6f}, max={log_depth.max():.6f}, mean={log_depth.mean():.6f}")
    
    # Normalize depth to [0, 1] using min/max (same as gsplat render_traj)
    depth_min = depth_vis[valid_mask].min()
    depth_max = depth_vis[valid_mask].max()
    
    if debug_label:
        print(f"      Normalization range: [{depth_min:.6f}, {depth_max:.6f}], range={depth_max-depth_min:.6f}")
    
    if depth_max <= depth_min:
        if debug_label:
            print(f"      [WARNING] depth_max <= depth_min, adjusting...")
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    
    # Normalize depth to [0, 1]
    depth_norm = np.clip((depth_vis - depth_min) / (depth_max - depth_min + 1e-10), 0.0, 1.0)
    
    if debug_label:
        norm_valid = depth_norm[valid_mask]
        print(f"      Normalized depth: min={norm_valid.min():.6f}, max={norm_valid.max():.6f}, mean={norm_valid.mean():.6f}")
        print(f"      Pixels at bounds: min_bound={np.sum(norm_valid <= 0.01)}, max_bound={np.sum(norm_valid >= 0.99)}")
    
    # Convert to grayscale (repeat 3 channels) and scale to [0, 255]
    depth_gray = (depth_norm * 255.0).astype(np.uint8)
    depth_gray = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)  # [H, W, 3]
    
    # Set invalid pixels to black
    depth_gray[~valid_mask] = 0
    
    if debug_label:
        gray_valid = depth_gray[valid_mask, 0]  # Use first channel
        print(f"      Output grayscale: min={gray_valid.min()}, max={gray_valid.max()}, mean={gray_valid.mean():.1f}")
    
    return depth_gray


def visualize_depth_map_colored(
    depth: np.ndarray,
    cmap: str = "Spectral",
    percentile: float = 2.0,
    use_disparity: bool = True,  # Changed default to True (DA3 style)
    color_space: str = "linear",
    debug_label: str = "",
) -> np.ndarray:
    """Convert depth map to colored visualization (depth-anything-3 style).
    
    This matches the depth visualization used in depth-anything-3:
    - Convert depth to disparity (1/depth) by default to handle uneven depth distribution
    - Use percentile-based min/max normalization to avoid extreme values
    - Apply colormap (default: Spectral)
    - Invert so near objects are red, far objects are blue
    
    Args:
        depth: [H, W] float32 depth map
        cmap: Matplotlib colormap name (default: "Spectral")
        percentile: Percentile for min/max computation (default: 2.0)
        use_disparity: Whether to convert depth to disparity (1/depth) before visualization.
                      If True, near objects will have higher values (better for observing close details).
                      This is the default DA3 behavior and helps with uneven depth distribution.
                      (default: True)
        color_space: "linear" for linear depth, "log" for log depth visualization (default: "linear")
        debug_label: Optional label for debug output
    
    Returns:
        [H, W, 3] uint8 colored depth visualization
    """
    depth = depth.copy()
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # No valid depth, return black image
        h, w = depth.shape
        if debug_label:
            print(f"    [Visualize Debug] {debug_label} (colored): No valid depth values!")
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Debug: Print input depth statistics
    valid_depth_orig = depth[valid_mask]
    if debug_label:
        print(f"    [Visualize Debug] {debug_label} (colored) - Input depth:")
        print(f"      Valid pixels: {np.sum(valid_mask)}/{depth.size} ({100*np.sum(valid_mask)/depth.size:.1f}%)")
        print(f"      Raw depth: min={valid_depth_orig.min():.6f}, max={valid_depth_orig.max():.6f}, mean={valid_depth_orig.mean():.6f}, std={valid_depth_orig.std():.6f}")
    
    # Convert depth to disparity (1/depth) - DA3 default behavior
    # This helps with uneven depth distribution by making near objects have higher values
    if use_disparity:
        depth[valid_mask] = 1.0 / depth[valid_mask]
        if debug_label:
            disparity = depth[valid_mask]
            print(f"      After disparity (1/depth): min={disparity.min():.6f}, max={disparity.max():.6f}, mean={disparity.mean():.6f}, std={disparity.std():.6f}")
    
    # Apply color space transformation
    if color_space == "log":
        # Log depth space (similar to depth-anything-3 tensor visualization)
        depth[valid_mask] = np.log(depth[valid_mask])
        if debug_label:
            log_depth = depth[valid_mask]
            print(f"      After log transform: min={log_depth.min():.6f}, max={log_depth.max():.6f}, mean={log_depth.mean():.6f}")
    
    # Compute min/max using percentiles (DA3 style: avoid extreme values)
    valid_depth = depth[valid_mask]
    if len(valid_depth) <= 10:
        # Not enough valid points, use min/max
        depth_min = valid_depth.min() if len(valid_depth) > 0 else 0.0
        depth_max = valid_depth.max() if len(valid_depth) > 0 else 0.0
        if debug_label:
            print(f"      Using min/max (too few points): [{depth_min:.6f}, {depth_max:.6f}]")
    else:
        # Use percentiles to avoid extreme values affecting visualization
        depth_min = np.percentile(valid_depth, percentile)
        depth_max = np.percentile(valid_depth, 100.0 - percentile)
        if debug_label:
            print(f"      Using {percentile}th percentile: min={depth_min:.6f}, max={depth_max:.6f}, range={depth_max-depth_min:.6f}")
            print(f"      Full range: [{valid_depth.min():.6f}, {valid_depth.max():.6f}]")
    
    # Ensure depth_min < depth_max
    if depth_min >= depth_max:
        if debug_label:
            print(f"      [WARNING] depth_min >= depth_max, adjusting...")
        if depth_min == depth_max:
            # All values are the same, add small epsilon
            depth_min = depth_min - 1e-6
            depth_max = depth_max + 1e-6
        else:
            # Swap if min > max (shouldn't happen, but safety check)
            depth_min, depth_max = depth_max, depth_min
    
    # Normalize to [0, 1] with clipping
    depth_range = depth_max - depth_min
    if depth_range < 1e-10:
        # Range is too small, use uniform mapping
        if debug_label:
            print(f"      [WARNING] Depth range too small ({depth_range:.2e}), using uniform mapping")
        depth_norm = np.ones_like(depth, dtype=np.float32) * 0.5
        depth_norm[~valid_mask] = 0.0
    else:
        depth_norm = ((depth - depth_min) / depth_range).clip(0.0, 1.0)
        depth_norm[~valid_mask] = 0.0
        if debug_label:
            norm_valid = depth_norm[valid_mask]
            print(f"      Normalized depth: min={norm_valid.min():.6f}, max={norm_valid.max():.6f}, mean={norm_valid.mean():.6f}")
            print(f"      Pixels at bounds: min_bound={np.sum(norm_valid <= 0.01)}, max_bound={np.sum(norm_valid >= 0.99)}")
    
    # Invert so near objects are red, far objects are blue (for Spectral colormap)
    # When use_disparity=True, near objects have higher disparity values, so we invert
    # to make near objects appear red (high colormap value) and far objects blue (low colormap value)
    depth_norm = 1.0 - depth_norm
    if debug_label:
        norm_valid_inv = depth_norm[valid_mask]
        print(f"      After inversion: min={norm_valid_inv.min():.6f}, max={norm_valid_inv.max():.6f}, mean={norm_valid_inv.mean():.6f}")
    
    # Apply colormap
    cm = matplotlib.colormaps[cmap]
    img_colored = cm(depth_norm[None], bytes=False)[:, :, :, 0:3]  # [1, H, W, 3], values in [0, 1]
    img_colored = (img_colored[0] * 255.0).astype(np.uint8)  # [H, W, 3]
    
    # Set invalid pixels to black
    img_colored[~valid_mask] = 0
    
    if debug_label:
        colored_valid = img_colored[valid_mask]
        print(f"      Output RGB: min=[{colored_valid.min(axis=0)}], max=[{colored_valid.max(axis=0)}], mean=[{colored_valid.mean(axis=0).astype(int)}]")
    
    return img_colored


def visualize_depth_map(
    depth: np.ndarray,
    mode: str = "grayscale",
    cmap: str = "Spectral",
    percentile: float = 2.0,
    use_disparity: bool = True,  # Changed default to True (DA3 style)
    color_space: str = "linear",
    debug_label: str = "",
) -> np.ndarray:
    """Convert depth map to visualization (grayscale or colored).
    
    Args:
        depth: [H, W] float32 depth map
        mode: "grayscale" or "colored" (default: "grayscale")
        cmap: Matplotlib colormap name (only used when mode="colored", default: "Spectral")
        percentile: Percentile for min/max computation (only used when mode="colored", default: 2.0)
        use_disparity: Whether to convert depth to disparity (1/depth) before visualization
                      (only used when mode="colored", default: True, DA3 style)
        color_space: "linear" for linear depth, "log" for log depth visualization (default: "linear")
        debug_label: Optional label for debug output
    
    Returns:
        [H, W, 3] uint8 depth visualization
    """
    if mode == "grayscale":
        return visualize_depth_map_gsplat_style(depth, color_space=color_space, debug_label=debug_label)
    elif mode == "colored":
        return visualize_depth_map_colored(depth, cmap=cmap, percentile=percentile, use_disparity=use_disparity, color_space=color_space, debug_label=debug_label)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'grayscale' or 'colored'.")


def create_cubemap_grid(
    cube_faces: Dict[str, np.ndarray],
    cube_depths: Dict[str, np.ndarray],
    face_order: List[str] = None,
    depth_mode: str = "grayscale",
    depth_cmap: str = "Spectral",
    depth_percentile: float = 2.0,
    use_disparity: bool = True,  # Changed default to True (DA3 style)
    color_space: str = "linear",
) -> np.ndarray:
    """Create a 2-row x 6-column grid image showing RGB and depth for each cubemap face.
    
    Args:
        cube_faces: Dict with keys ['posx', 'negx', 'posy', 'negy', 'posz', 'negz'],
                    each value is [H, W, 3] uint8 RGB image
        cube_depths: Dict with same keys, each value is [H, W] float32 depth map
        face_order: Order of faces to display. Default: ['posz', 'posx', 'negz', 'negx', 'posy', 'negy']
                    (front, right, back, left, top, bottom)
        depth_mode: "grayscale" or "colored" (default: "grayscale")
        depth_cmap: Matplotlib colormap name (only used when depth_mode="colored", default: "Spectral")
        depth_percentile: Percentile for min/max computation (only used when depth_mode="colored", default: 2.0)
        use_disparity: Whether to convert depth to disparity (1/depth) before visualization
                      (only used when depth_mode="colored", default: True, DA3 style)
        color_space: "linear" for linear depth, "log" for log depth visualization (default: "linear")
    
    Returns:
        [2*H, 6*W, 3] uint8 grid image
    """
    if face_order is None:
        # Default order: front, right, back, left, top, bottom
        face_order = ['posz', 'posx', 'negz', 'negx', 'posy', 'negy']
    
    # Get face size from first face
    first_face = cube_faces[face_order[0]]
    face_h, face_w = first_face.shape[:2]
    
    # Create grid: 2 rows (RGB, depth) x 6 columns (faces)
    grid = np.zeros((2 * face_h, 6 * face_w, 3), dtype=np.uint8)
    
    # First row: RGB images
    for col, face_name in enumerate(face_order):
        if face_name in cube_faces:
            grid[0:face_h, col*face_w:(col+1)*face_w, :] = cube_faces[face_name]
    
    # Second row: Depth images
    for col, face_name in enumerate(face_order):
        if face_name in cube_depths:
            depth_face = cube_depths[face_name]
            
            # Debug: Print depth statistics (before transformation)
            valid_depth = depth_face[depth_face > 0]
            if len(valid_depth) > 0:
                # 计算更多统计信息（原始深度值）
                p1, p5, p10, p90, p95, p99 = np.percentile(valid_depth, [1, 5, 10, 90, 95, 99])
                print(f"    [Depth Debug] {face_name} (raw depth): "
                      f"min={valid_depth.min():.3f}, max={valid_depth.max():.3f}, "
                      f"mean={valid_depth.mean():.3f}, std={valid_depth.std():.3f}")
                print(f"    [Depth Debug] {face_name} percentiles: "
                      f"1%={p1:.3f}, 5%={p5:.3f}, 10%={p10:.3f}, "
                      f"90%={p90:.3f}, 95%={p95:.3f}, 99%={p99:.3f}")
                
                # 检查深度分布是否过于集中
                depth_range = valid_depth.max() - valid_depth.min()
                if depth_range < 0.1:
                    print(f"    [WARNING] {face_name}: Depth range ({depth_range:.3f}) is very small! "
                          f"Visualization may appear solid color.")
                
                # 如果使用 disparity，显示转换后的统计信息
                if use_disparity and depth_mode == "colored":
                    disparity = 1.0 / valid_depth
                    disp_p1, disp_p5, disp_p10, disp_p90, disp_p95, disp_p99 = np.percentile(disparity, [1, 5, 10, 90, 95, 99])
                    print(f"    [Depth Debug] {face_name} (disparity=1/depth): "
                          f"min={disparity.min():.3f}, max={disparity.max():.3f}, "
                          f"mean={disparity.mean():.3f}, std={disparity.std():.3f}")
                    print(f"    [Depth Debug] {face_name} disparity percentiles: "
                          f"1%={disp_p1:.3f}, 5%={disp_p5:.3f}, 10%={disp_p10:.3f}, "
                          f"90%={disp_p90:.3f}, 95%={disp_p95:.3f}, 99%={disp_p99:.3f}")
                    disp_range = disparity.max() - disparity.min()
                    if disp_range < 0.1:
                        print(f"    [WARNING] {face_name}: Disparity range ({disp_range:.3f}) is very small!")
                
                if color_space == "log":
                    log_depth = np.log(valid_depth)
                    log_range = log_depth.max() - log_depth.min()
                    print(f"    [Depth Debug] {face_name} (log space): "
                          f"min={log_depth.min():.3f}, max={log_depth.max():.3f}, "
                          f"range={log_range:.3f}")
                    if log_range < 0.1:
                        print(f"    [WARNING] {face_name}: Log depth range ({log_range:.3f}) is very small!")
            else:
                print(f"    [Depth Debug] {face_name}: No valid depth values!")
            
            depth_vis = visualize_depth_map(
                depth_face,
                mode=depth_mode,
                cmap=depth_cmap,
                percentile=depth_percentile,
                use_disparity=use_disparity,
                color_space=color_space,
                debug_label=f"cubemap_{face_name}",
            )
            grid[face_h:2*face_h, col*face_w:(col+1)*face_w, :] = depth_vis
    
    return grid


def create_pano_combined(
    pano: np.ndarray,
    pano_depth: np.ndarray,
    depth_mode: str = "grayscale",
    depth_cmap: str = "Spectral",
    depth_percentile: float = 2.0,
    use_disparity: bool = True,  # Changed default to True (DA3 style)
    color_space: str = "linear",
) -> np.ndarray:
    """Create a vertically stacked image with RGB panorama on top and depth panorama on bottom.
    
    Args:
        pano: [H, W, 3] uint8 RGB panorama
        pano_depth: [H, W] float32 depth panorama
        depth_mode: "grayscale" or "colored" (default: "grayscale")
        depth_cmap: Matplotlib colormap name (only used when depth_mode="colored", default: "Spectral")
        depth_percentile: Percentile for min/max computation (only used when depth_mode="colored", default: 2.0)
        use_disparity: Whether to convert depth to disparity (1/depth) before visualization
                      (only used when depth_mode="colored", default: True, DA3 style)
        color_space: "linear" for linear depth, "log" for log depth visualization (default: "linear")
    
    Returns:
        [2*H, W, 3] uint8 combined image
    """
    pano_h, pano_w = pano.shape[:2]
    
    # Debug: Print panorama depth statistics before visualization
    valid_pano_mask = pano_depth > 0
    if np.any(valid_pano_mask):
        valid_pano_depth = pano_depth[valid_pano_mask]
        print(f"    [Pano Combined Debug] Input panorama depth:")
        print(f"      Valid pixels: {np.sum(valid_pano_mask)}/{pano_depth.size} ({100*np.sum(valid_pano_mask)/pano_depth.size:.1f}%)")
        print(f"      Depth stats: min={valid_pano_depth.min():.6f}, max={valid_pano_depth.max():.6f}, mean={valid_pano_depth.mean():.6f}, std={valid_pano_depth.std():.6f}")
    
    # Convert depth to visualization
    depth_vis = visualize_depth_map(
        pano_depth,
        mode=depth_mode,
        cmap=depth_cmap,
        percentile=depth_percentile,
        use_disparity=use_disparity,
        color_space=color_space,
        debug_label="pano_combined",
    )
    
    # Stack vertically: RGB on top, depth on bottom
    combined = np.zeros((2 * pano_h, pano_w, 3), dtype=np.uint8)
    combined[0:pano_h, :, :] = pano
    combined[pano_h:2*pano_h, :, :] = depth_vis
    
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export gsplat cubemap + panorama for selected views, "
            "in a gs_pandepth-style manner."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Path to YAML config. If omitted, use default pandepth_config.yaml "
            "next to this script."
        ),
    )
    return parser.parse_args()


def main_entry() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
    else:
        cfg_path = script_dir / "pandepth_config.yaml"

    cfg = load_yaml_config(cfg_path)

    data_dir = cfg.get("data_dir")
    if not data_dir:
        raise ValueError("Config must contain 'data_dir'.")
    data_dir = str(Path(data_dir).expanduser())

    result_dir = cfg.get("result_dir")
    if not result_dir:
        raise ValueError("Config must contain 'result_dir'.")
    result_dir = str(Path(result_dir).expanduser())

    ckpt_path = cfg.get("ckpt")
    if not ckpt_path:
        raise ValueError("Config must contain 'ckpt' (path to gsplat checkpoint).")
    ckpt_path = Path(ckpt_path).expanduser().resolve()

    # 新的配置方式：使用 pano_front_camera_index 作为前向参考相机
    pano_cameras = cfg.get("pano_cameras", None)  # 例如 [0, 1, 15] - 要导出的全景图编号
    pano_front_camera_index = cfg.get("pano_front_camera_index", None)  # 例如 12 - 前向参考相机编号

    # 兼容旧配置：如果提供了 pandepth_indices，则使用旧方式
    pandepth_indices: List[int] = cfg.get("pandepth_indices") or []
    use_legacy_mode = len(pandepth_indices) > 0

    cube_face_size: int = int(cfg.get("cube_face_size", 1024))
    pano_size = cfg.get("pano_size", None)
    if pano_size is None:
        pano_h, pano_w = cube_face_size, cube_face_size * 2
    else:
        if not (isinstance(pano_size, (list, tuple)) and len(pano_size) == 2):
            raise ValueError("pano_size must be [H, W].")
        pano_h, pano_w = int(pano_size[0]), int(pano_size[1])

    near_plane = float(cfg.get("near_plane", 0.01))
    far_plane = float(cfg.get("far_plane", 1.0e10))
    sh_degree = int(cfg.get("sh_degree", 3))  # Spherical harmonics degree (default 3)

    # Depth visualization settings
    depth_mode = str(cfg.get("depth_mode", "grayscale")).lower()  # "grayscale" or "colored"
    if depth_mode not in ["grayscale", "colored"]:
        raise ValueError(f"Invalid depth_mode: {depth_mode}. Must be 'grayscale' or 'colored'.")
    depth_cmap = str(cfg.get("depth_cmap", "Spectral"))  # Matplotlib colormap name
    depth_percentile = float(cfg.get("depth_percentile", 2.0))  # Percentile for min/max computation
    use_disparity = bool(cfg.get("use_disparity", True))  # Whether to use disparity (1/depth) for visualization (default: True, DA3 style)
    
    # Depth type and color space settings
    depth_type = str(cfg.get("depth_type", "expected")).lower()  # "expected" (ED) or "accumulated" (D)
    if depth_type not in ["expected", "accumulated"]:
        raise ValueError(f"Invalid depth_type: {depth_type}. Must be 'expected' or 'accumulated'.")
    depth_color_space = str(cfg.get("depth_color_space", "linear")).lower()  # "linear" or "log"
    if depth_color_space not in ["linear", "log"]:
        raise ValueError(f"Invalid depth_color_space: {depth_color_space}. Must be 'linear' or 'log'.")

    factor = int(cfg.get("data_factor", 1))
    normalize_world_space = bool(cfg.get("normalize_world_space", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pandepth_exporter] Using device: {device}")
    print(f"[pandepth_exporter] Depth type: {depth_type} ({'ED (Expected Depth)' if depth_type == 'expected' else 'D (Accumulated Depth)'})")
    print(f"[pandepth_exporter] Depth color space: {depth_color_space}")
    print(f"[pandepth_exporter] Depth visualization mode: {depth_mode}")

    # 1. Load splats from checkpoint.
    splats = load_splats_from_ckpt(ckpt_path, device=device)
    print(
        f"[pandepth_exporter] Loaded splats from {ckpt_path}, "
        f"N={splats['means'].shape[0]}"
    )

    # 2. Parse COLMAP dataset (same logic as training).
    parser = Parser(
        data_dir=data_dir,
        factor=factor,
        normalize=normalize_world_space,
        test_every=8,
        load_exposure=False,
        pano_image_indices=None,  # 不在这里过滤，我们要自己选择
    )
    camtoworlds = parser.camtoworlds  # [M, 4, 4], numpy
    image_names = parser.image_names  # List[str], 例如 ["pano_camera0/xxx.jpg", ...]

    num_views = camtoworlds.shape[0]
    print(f"[pandepth_exporter] Dataset has {num_views} views.")

    # 3. 根据配置选择视角
    import re
    cam_pattern = re.compile(r"pano_camera(\d+)/")

    selected_c2w_list: List[Tuple[int, np.ndarray, str]] = []  # [(global_idx, c2w, cam_name), ...]

    if use_legacy_mode:
        # 旧方式：使用 pandepth_indices（排序后的索引）
        print("[pandepth_exporter] Using legacy mode: pandepth_indices")
        valid_indices: List[int] = []
        for idx in pandepth_indices:
            if 0 <= idx < num_views:
                valid_indices.append(idx)
            else:
                print(
                    f"  [Warning] pandepth index {idx} is out of range (0..{num_views-1}), skipped."
                )
        if not valid_indices:
            raise ValueError("All pandepth_indices are out of range.")
        for idx in valid_indices:
            selected_c2w_list.append((idx, camtoworlds[idx], f"view_{idx:04d}"))
    else:
        # 新方式：使用 pano_front_camera_index 作为前向参考
        # 对于 pano_cameras 中的每个相机编号 i，使用 pano_front_camera_index 的第 i 张图片的外参
        if pano_cameras is None or pano_front_camera_index is None:
            raise ValueError(
                "Config must provide either 'pandepth_indices' (legacy) or "
                "'pano_cameras' + 'pano_front_camera_index' (new mode)."
            )

        # 按相机分组
        per_cam: Dict[int, List[Tuple[int, str]]] = {}  # {camera_num: [(global_idx, image_name), ...]}
        for g_idx, name in enumerate(image_names):
            m = cam_pattern.match(name)
            if not m:
                continue  # 跳过非 pano_camera 格式的图片
            cam_num = int(m.group(1))
            per_cam.setdefault(cam_num, []).append((g_idx, name))

        # 对每个相机内的图片按文件名排序
        for cam_num in per_cam.keys():
            per_cam[cam_num].sort(key=lambda x: x[1])  # 按 image_name 排序

        # 检查前向参考相机是否存在
        if pano_front_camera_index not in per_cam:
            raise ValueError(
                f"pano_front_camera_index={pano_front_camera_index} (pano_camera{pano_front_camera_index}) "
                f"not found in dataset. Available cameras: {sorted(per_cam.keys())}"
            )

        front_cam_images = per_cam[pano_front_camera_index]
        print(
            f"[pandepth_exporter] Using pano_camera{pano_front_camera_index} as front reference "
            f"({len(front_cam_images)} images available)"
        )

        # 对于每个要导出的全景图，使用前向参考相机中对应索引的图片外参
        for pano_idx in pano_cameras:
            if pano_idx >= len(front_cam_images):
                print(
                    f"  [Warning] pano_camera{pano_front_camera_index} only has {len(front_cam_images)} images, "
                    f"but requested index {pano_idx} for pano_camera{pano_idx}, skipped."
                )
                continue
            g_idx, img_name = front_cam_images[pano_idx]
            c2w = camtoworlds[g_idx]
            selected_c2w_list.append((g_idx, c2w, f"pano_camera{pano_idx:02d}"))
            print(
                f"  [Info] pano_camera{pano_idx:02d} -> using pano_camera{pano_front_camera_index} "
                f"image #{pano_idx} ({img_name})"
            )

    if not selected_c2w_list:
        raise ValueError("No valid views selected for export.")

    print(f"[pandepth_exporter] Selected {len(selected_c2w_list)} views for export.")

    os.makedirs(result_dir, exist_ok=True)

    # 4. For each selected view, render cubemap and panorama.
    meta: Dict[str, Dict] = {}
    total_views = len(selected_c2w_list)
    for view_num, (view_idx, c2w, view_tag) in enumerate(selected_c2w_list, 1):
        out_dir = os.path.join(result_dir, view_tag)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[pandepth_exporter] [{view_num}/{total_views}] Rendering {view_tag} (global_idx={view_idx})")
        print(f"  Output directory: {out_dir}")

        try:
            # Render RGB and depth cubemaps
            print(f"  [Render] Rendering cubemap with depth_type={depth_type}, near={near_plane}, far={far_plane}")
            cube_faces, cube_depths = render_cubemap_for_view(
                splats=splats,
                c2w_center=c2w,
                face_size=cube_face_size,
                near_plane=near_plane,
                far_plane=far_plane,
                device=device,
                sh_degree=sh_degree,
                render_depth=True,
                depth_type=depth_type,
            )
            print(f"  ✓ Cubemap rendered: {len(cube_faces)} RGB faces, {len(cube_depths)} depth faces ({depth_type} depth)")

            # Convert to panorama (RGB and depth).
            print(f"  [Convert] Converting cubemap to equirectangular panorama ({pano_h}x{pano_w})")
            pano, pano_depth = cubemap_to_equirect(
                cube_faces, 
                pano_h=pano_h, 
                pano_w=pano_w,
                cube_depths=cube_depths,
            )
            print(f"  ✓ Panorama converted")
            
            # Create and save cubemap grid (2 rows x 6 columns: RGB and depth for each face)
            # Face order: front (posz), right (posx), back (negz), left (negx), top (posy), bottom (negy)
            print(f"  [Visualize] Creating cubemap grid with depth_mode={depth_mode}, use_disparity={use_disparity}, color_space={depth_color_space}")
            cubemap_grid = create_cubemap_grid(
                cube_faces, 
                cube_depths,
                depth_mode=depth_mode,
                depth_cmap=depth_cmap,
                depth_percentile=depth_percentile,
                use_disparity=use_disparity,
                color_space=depth_color_space,
            )
            print(f"  ✓ Cubemap grid created")
            cubemap_grid_path = os.path.join(out_dir, f"{view_tag}_cubemap_grid.png")
            imageio.imwrite(cubemap_grid_path, cubemap_grid)
            print(f"  ✓ Cubemap grid saved: {cubemap_grid_path}")
            
            # Create and save combined panorama (RGB on top, depth on bottom)
            if pano_depth is not None:
                print(f"  [Visualize] Creating combined panorama with depth_mode={depth_mode}, use_disparity={use_disparity}, color_space={depth_color_space}")
                pano_combined = create_pano_combined(
                    pano, 
                    pano_depth,
                    depth_mode=depth_mode,
                    depth_cmap=depth_cmap,
                    depth_percentile=depth_percentile,
                    use_disparity=use_disparity,
                    color_space=depth_color_space,
                )
                print(f"  ✓ Combined panorama created")
                pano_combined_path = os.path.join(out_dir, f"{view_tag}_pano_combined.png")
                imageio.imwrite(pano_combined_path, pano_combined)
                print(f"  ✓ Combined panorama saved: {pano_combined_path}")
        except Exception as e:
            print(f"  ✗ Error rendering {view_tag}: {e}")
            import traceback
            traceback.print_exc()
            continue

        meta[view_tag] = {
            "global_index": int(view_idx),
            "image_name": image_names[view_idx] if view_idx < len(image_names) else "unknown",
            "cube_face_size": int(cube_face_size),
            "pano_size": [int(pano_h), int(pano_w)],
        }

    # 4. Save simple metadata JSON.
    meta_path = os.path.join(result_dir, "pandepth_export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[pandepth_exporter] Done. Metadata saved to {meta_path}")


if __name__ == "__main__":
    main_entry()

