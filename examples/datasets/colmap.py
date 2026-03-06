import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from exif import compute_exposure_from_exif
from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
        pano_image_indices: Optional[List[int]] = None,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure
        self.pano_image_indices = pano_image_indices

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Optional: restrict to specific per-camera image indices for pano_camera rigs.
        # pano_image_indices is interpreted as a 1-based index list within each
        # pano_cameraXX/ directory after sorting filenames.
        if self.pano_image_indices is not None:
            # Validate indices are positive.
            valid_p = [p for p in self.pano_image_indices if p >= 1]
            if len(valid_p) != len(self.pano_image_indices):
                raise ValueError(
                    f"pano_image_indices must be 1-based positive integers, got {self.pano_image_indices}"
                )

            import re

            cam_pattern = re.compile(r"(pano_camera\d+)/")
            # Group global indices by pano_camera directory.
            per_cam: Dict[str, List[int]] = {}
            for g_idx, name in enumerate(image_names):
                m = cam_pattern.match(name)
                if not m:
                    # Not a pano_camera-style image; currently drop it when filtering.
                    continue
                cam_dir = m.group(1)
                per_cam.setdefault(cam_dir, []).append(g_idx)

            keep_global_indices: List[int] = []
            for cam_dir, g_indices in per_cam.items():
                # Sort by filename within this camera.
                g_indices_sorted = sorted(g_indices, key=lambda i: image_names[i])
                num_imgs = len(g_indices_sorted)
                # Convert 1-based indices to 0-based within this camera.
                for p in valid_p:
                    if p <= num_imgs:
                        keep_global_indices.append(g_indices_sorted[p - 1])

            if not keep_global_indices:
                raise ValueError(
                    f"pano_image_indices={self.pano_image_indices} resulted in zero selected images."
                )

            keep_global_indices = sorted(set(keep_global_indices))

            # Apply selection to per-image arrays.
            image_names = [image_names[i] for i in keep_global_indices]
            camtoworlds = camtoworlds[keep_global_indices]
            camera_ids = [camera_ids[i] for i in keep_global_indices]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # Create 0-based contiguous camera indices from COLMAP camera_ids.
        # This is useful for camera-based embeddings/modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Load EXIF exposure data if requested.
        # Always read from original (non-downscaled) images since PNG doesn't support EXIF.
        if load_exposure:
            exposure_values: List[Optional[float]] = []
            for image_name in tqdm(image_names, desc="Loading EXIF exposure"):
                original_path = Path(colmap_image_dir) / image_name
                exposure_values.append(compute_exposure_from_exif(original_path))

            # Compute mean across all valid exposures and subtract
            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                print(
                    f"[Parser] Loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )
            else:
                self.exposure_values = [None] * len(exposure_values)
                print("[Parser] No valid EXIF exposure data found in any image.")
        else:
            self.exposure_values = [None] * len(image_paths)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        use_external_depth: bool = False,
        external_depth_dir: Optional[str] = None,
        external_depth_type: str = "img",
        external_depth_path: Optional[str] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.use_external_depth = use_external_depth
        self.external_depth_dir = external_depth_dir
        self.external_depth_type = external_depth_type
        self.external_depth_path = external_depth_path

        if self.load_depths and self.use_external_depth:
            if self.external_depth_type == "img" and self.external_depth_dir is None:
                raise ValueError(
                    "external_depth_dir must be provided when use_external_depth is True "
                    "and external_depth_type is 'img'."
                )
            if self.external_depth_type == "ply" and self.external_depth_path is None:
                raise ValueError(
                    "external_depth_path must be provided when use_external_depth is True "
                    "and external_depth_type is 'ply'."
                )
        # Global indices after any Parser-level filtering (e.g. pano_image_indices).
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

        # Debug info: how many images actually used for this split.
        # This number already reflects pano_image_indices filtering in Parser
        # and the train/val split by test_every.
        print(
            f"[Dataset] split='{self.split}': "
            f"{len(self.indices)} images used out of {len(self.parser.image_names)} "
            f"Parser images (test_every={self.parser.test_every})."
        )

        # Pre-load fused point cloud when using PLY-based external depth.
        self._ply_points_world: Optional[np.ndarray] = None
        if self.load_depths and self.use_external_depth and self.external_depth_type == "ply":
            from pathlib import Path

            ply_path = Path(self.external_depth_path).expanduser()
            if not ply_path.is_file():
                raise FileNotFoundError(
                    f"External depth PLY file not found: {ply_path}"
                )
            try:
                import open3d as o3d  # type: ignore
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "PLY-based external depth requires the 'open3d' package. "
                    "Please install it via 'pip install open3d'."
                ) from e

            pcd = o3d.io.read_point_cloud(str(ply_path))
            if len(pcd.points) == 0:
                raise ValueError(f"No points found in external depth PLY: {ply_path}")
            self._ply_points_world = np.asarray(pcd.points, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_depths:
            if self.use_external_depth:
                if self.external_depth_type == "img":
                    # Load external dense depth map aligned with the (possibly cropped) image.
                    # Depth files are expected to mirror COLMAP image relative paths:
                    #   images/pano_camera0/xxx.jpg -> external_depth_dir/pano_camera0/xxx.npy
                    image_name = self.parser.image_names[index]
                    rel_path = image_name
                    stem, _ = os.path.splitext(rel_path)
                    # Try .npy then .npz
                    depth_path_npy = os.path.join(self.external_depth_dir, stem + ".npy")
                    depth_path_npz = os.path.join(self.external_depth_dir, stem + ".npz")
                    depth = None
                    if os.path.isfile(depth_path_npy):
                        depth = np.load(depth_path_npy)
                    elif os.path.isfile(depth_path_npz):
                        npz = np.load(depth_path_npz)
                        # heuristics: use "depth" key if present, else first array
                        if isinstance(npz, np.lib.npyio.NpzFile):
                            if "depth" in npz.files:
                                depth = npz["depth"]
                            else:
                                depth = npz[npz.files[0]]
                        else:
                            depth = npz
                    else:
                        raise FileNotFoundError(
                            f"External depth map not found for image '{image_name}'. "
                            f"Tried: {depth_path_npy}, {depth_path_npz}"
                        )

                    # Ensure depth is HxW and matches current image size; resize if needed.
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    if depth.shape[:2] != image.shape[:2]:
                        depth = cv2.resize(
                            depth.astype(np.float32),
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    data["depth_map"] = torch.from_numpy(depth.astype(np.float32))
                elif self.external_depth_type == "ply":
                    # Reproject fused PLY point cloud to obtain a dense-ish depth map.
                    assert (
                        self._ply_points_world is not None
                    ), "PLY points not loaded for external depth."
                    worldtocams = np.linalg.inv(camtoworlds)  # [4, 4]
                    R = worldtocams[:3, :3]
                    t = worldtocams[:3, 3:4]
                    pts_world = self._ply_points_world  # [N, 3]
                    pts_cam = (R @ pts_world.T + t).T  # [N, 3]
                    z = pts_cam[:, 2]
                    # keep only points in front of the camera
                    front = z > 0
                    pts_cam = pts_cam[front]
                    z = z[front]
                    if pts_cam.shape[0] > 0:
                        pts_proj = (K @ pts_cam.T).T  # [N, 3]
                        x = pts_proj[:, 0] / pts_proj[:, 2]
                        y = pts_proj[:, 1] / pts_proj[:, 2]
                        H, W = image.shape[:2]
                        x_int = x.astype(np.int32)
                        y_int = y.astype(np.int32)
                        inside = (
                            (x_int >= 0)
                            & (x_int < W)
                            & (y_int >= 0)
                            & (y_int < H)
                        )
                        x_int = x_int[inside]
                        y_int = y_int[inside]
                        z_inside = z[inside]
                        depth_map = np.zeros((H, W), dtype=np.float32)
                        # For each pixel, keep the nearest depth.
                        for xi, yi, zi in zip(x_int, y_int, z_inside):
                            current = depth_map[yi, xi]
                            if current == 0.0 or zi < current:
                                depth_map[yi, xi] = zi
                        data["depth_map"] = torch.from_numpy(depth_map)
                    else:
                        # No points in front of camera; provide an all-zero depth map.
                        depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                        data["depth_map"] = torch.from_numpy(depth_map)
                else:
                    raise ValueError(
                        f"Unknown external_depth_type: {self.external_depth_type}"
                    )
            else:
                # projected points to image plane to get depths from COLMAP sparse points
                worldtocams = np.linalg.inv(camtoworlds)
                image_name = self.parser.image_names[index]
                point_indices = self.parser.point_indices[image_name]
                points_world = self.parser.points[point_indices]
                points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
                depths = points_cam[:, 2]  # (M,)
                # filter out points outside the image
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
                data["points"] = torch.from_numpy(points).float()
                data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=True,
        test_every=8,
        load_exposure=False,
        pano_image_indices=None,
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
