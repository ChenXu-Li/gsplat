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
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Render 6 cubemap faces (RGB and optionally depth) around a given center camera pose.
    
    Returns:
        face_images: Dict[str, np.ndarray] - RGB images for each face
        face_depths: Dict[str, np.ndarray] - Depth images for each face (if render_depth=True)
    """
    c2w_faces_np, face_names = make_cube_camera_matrices(c2w_center)
    c2w_faces = torch.from_numpy(c2w_faces_np).to(device=device, dtype=torch.float32)
    viewmats = torch.linalg.inv(c2w_faces)  # [6, 4, 4] world-to-camera

    K_face = make_cube_intrinsics(face_size).to(device)
    Ks = K_face.unsqueeze(0).repeat(len(face_names), 1, 1)  # [6, 3, 3]

    render_mode = "RGB+ED" if render_depth else "RGB"
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
    colors = torch.clamp(colors, 0.0, 1.0)  # [6, H, W, 3] or [6, H, W, 4]

    face_images: Dict[str, np.ndarray] = {}
    face_depths: Dict[str, np.ndarray] = {}
    
    for i, name in enumerate(face_names):
        if render_depth:
            # Extract RGB and depth
            rgb = colors[i, ..., :3]  # [H, W, 3]
            depth = colors[i, ..., 3:4]  # [H, W, 1]
            img = (rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)
            depth_np = depth.detach().cpu().numpy().squeeze(-1)  # [H, W]
            face_images[name] = img
            face_depths[name] = depth_np
        else:
            img = (colors[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
            face_images[name] = img
    
    if render_depth:
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

    return pano, pano_depth


def visualize_depth_map_gsplat_style(
    depth: np.ndarray,
) -> np.ndarray:
    """Convert depth map to grayscale visualization (gsplat video style).
    
    This matches the depth visualization used in gsplat's render_traj:
    - Normalize depth to [0, 1] using min/max
    - Convert to grayscale (repeat 3 channels)
    
    Args:
        depth: [H, W] float32 depth map
    
    Returns:
        [H, W, 3] uint8 grayscale depth visualization
    """
    # Create valid mask (depth > 0)
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # No valid depth, return black image
        h, w = depth.shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize depth to [0, 1] using min/max (same as gsplat render_traj)
    depth_min = depth[valid_mask].min()
    depth_max = depth[valid_mask].max()
    
    if depth_max <= depth_min:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    
    # Normalize depth to [0, 1]
    depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-10), 0.0, 1.0)
    
    # Convert to grayscale (repeat 3 channels) and scale to [0, 255]
    depth_gray = (depth_norm * 255.0).astype(np.uint8)
    depth_gray = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)  # [H, W, 3]
    
    # Set invalid pixels to black
    depth_gray[~valid_mask] = 0
    
    return depth_gray


def visualize_depth_map_colored(
    depth: np.ndarray,
    cmap: str = "Spectral",
    percentile: float = 2.0,
) -> np.ndarray:
    """Convert depth map to colored visualization (depth-anything-3 style).
    
    This matches the depth visualization used in depth-anything-3:
    - Convert depth to disparity (1/depth)
    - Use percentile-based min/max normalization
    - Apply colormap (default: Spectral)
    - Invert so near objects are red, far objects are blue
    
    Args:
        depth: [H, W] float32 depth map
        cmap: Matplotlib colormap name (default: "Spectral")
        percentile: Percentile for min/max computation (default: 2.0)
    
    Returns:
        [H, W, 3] uint8 colored depth visualization
    """
    depth = depth.copy()
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # No valid depth, return black image
        h, w = depth.shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    # Convert depth to disparity (1/depth)
    depth[valid_mask] = 1.0 / depth[valid_mask]
    
    # Compute min/max using percentiles
    if valid_mask.sum() <= 10:
        depth_min = 0.0
        depth_max = 0.0
    else:
        depth_min = np.percentile(depth[valid_mask], percentile)
        depth_max = np.percentile(depth[valid_mask], 100.0 - percentile)
    
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    
    # Normalize to [0, 1]
    depth_norm = ((depth - depth_min) / (depth_max - depth_min + 1e-10)).clip(0, 1)
    
    # Invert so near objects are red, far objects are blue (for Spectral colormap)
    depth_norm = 1.0 - depth_norm
    
    # Apply colormap
    cm = matplotlib.colormaps[cmap]
    img_colored = cm(depth_norm[None], bytes=False)[:, :, :, 0:3]  # [1, H, W, 3], values in [0, 1]
    img_colored = (img_colored[0] * 255.0).astype(np.uint8)  # [H, W, 3]
    
    # Set invalid pixels to black
    img_colored[~valid_mask] = 0
    
    return img_colored


def visualize_depth_map(
    depth: np.ndarray,
    mode: str = "grayscale",
    cmap: str = "Spectral",
    percentile: float = 2.0,
) -> np.ndarray:
    """Convert depth map to visualization (grayscale or colored).
    
    Args:
        depth: [H, W] float32 depth map
        mode: "grayscale" or "colored" (default: "grayscale")
        cmap: Matplotlib colormap name (only used when mode="colored", default: "Spectral")
        percentile: Percentile for min/max computation (only used when mode="colored", default: 2.0)
    
    Returns:
        [H, W, 3] uint8 depth visualization
    """
    if mode == "grayscale":
        return visualize_depth_map_gsplat_style(depth)
    elif mode == "colored":
        return visualize_depth_map_colored(depth, cmap=cmap, percentile=percentile)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'grayscale' or 'colored'.")


def create_cubemap_grid(
    cube_faces: Dict[str, np.ndarray],
    cube_depths: Dict[str, np.ndarray],
    face_order: List[str] = None,
    depth_mode: str = "grayscale",
    depth_cmap: str = "Spectral",
    depth_percentile: float = 2.0,
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
            depth_vis = visualize_depth_map(
                cube_depths[face_name],
                mode=depth_mode,
                cmap=depth_cmap,
                percentile=depth_percentile,
            )
            grid[face_h:2*face_h, col*face_w:(col+1)*face_w, :] = depth_vis
    
    return grid


def create_pano_combined(
    pano: np.ndarray,
    pano_depth: np.ndarray,
    depth_mode: str = "grayscale",
    depth_cmap: str = "Spectral",
    depth_percentile: float = 2.0,
) -> np.ndarray:
    """Create a vertically stacked image with RGB panorama on top and depth panorama on bottom.
    
    Args:
        pano: [H, W, 3] uint8 RGB panorama
        pano_depth: [H, W] float32 depth panorama
        depth_mode: "grayscale" or "colored" (default: "grayscale")
        depth_cmap: Matplotlib colormap name (only used when depth_mode="colored", default: "Spectral")
        depth_percentile: Percentile for min/max computation (only used when depth_mode="colored", default: 2.0)
    
    Returns:
        [2*H, W, 3] uint8 combined image
    """
    pano_h, pano_w = pano.shape[:2]
    
    # Convert depth to visualization
    depth_vis = visualize_depth_map(
        pano_depth,
        mode=depth_mode,
        cmap=depth_cmap,
        percentile=depth_percentile,
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

    factor = int(cfg.get("data_factor", 1))
    normalize_world_space = bool(cfg.get("normalize_world_space", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pandepth_exporter] Using device: {device}")

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
            cube_faces, cube_depths = render_cubemap_for_view(
                splats=splats,
                c2w_center=c2w,
                face_size=cube_face_size,
                near_plane=near_plane,
                far_plane=far_plane,
                device=device,
                sh_degree=sh_degree,
                render_depth=True,
            )
            print(f"  ✓ Cubemap rendered: {len(cube_faces)} RGB faces, {len(cube_depths)} depth faces")

            # Convert to panorama (RGB and depth).
            pano, pano_depth = cubemap_to_equirect(
                cube_faces, 
                pano_h=pano_h, 
                pano_w=pano_w,
                cube_depths=cube_depths,
            )
            
            # Create and save cubemap grid (2 rows x 6 columns: RGB and depth for each face)
            # Face order: front (posz), right (posx), back (negz), left (negx), top (posy), bottom (negy)
            cubemap_grid = create_cubemap_grid(
                cube_faces, 
                cube_depths,
                depth_mode=depth_mode,
                depth_cmap=depth_cmap,
                depth_percentile=depth_percentile,
            )
            cubemap_grid_path = os.path.join(out_dir, f"{view_tag}_cubemap_grid.png")
            imageio.imwrite(cubemap_grid_path, cubemap_grid)
            print(f"  ✓ Cubemap grid saved: {cubemap_grid_path}")
            
            # Create and save combined panorama (RGB on top, depth on bottom)
            if pano_depth is not None:
                pano_combined = create_pano_combined(
                    pano, 
                    pano_depth,
                    depth_mode=depth_mode,
                    depth_cmap=depth_cmap,
                    depth_percentile=depth_percentile,
                )
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

