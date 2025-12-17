import os
import random
import math
from typing import Callable, Dict, List, Optional, Tuple

import os
import torch
import numpy as np
import trimesh
from skimage import measure
from DeepSDFTrainer import DeepSDF  
from torch.utils.data import Dataset, DataLoader
from DeepSDFTrainer import DeepSDF, DeepSDFTrainer

SDFCallable = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
SceneWithOperators = Dict[int, Tuple[SDFCallable, List[Tuple[float, float]]]]
Scenes = Dict[str, SceneWithOperators]


class SceneSDFDataset(Dataset):
    """
    Each item corresponds to one scene.

    Returns:
        scene_id : int
        points   : (N, D)
        sdf      : (N, 1)
    """

    def __init__(self, samples: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        self.samples = samples
        self.scene_ids = sorted(samples.keys())

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        pts, sdf = self.samples[sid]

        return (
            sid,
            torch.from_numpy(pts).float(),
            torch.from_numpy(sdf).float(),
        )


class Model:
    def __init__(
        self,
        base_directory: str,
        model_name: str,
        scenes: Scenes,
        domain_radius: float = 1.0,
        latent_dim: int = 256,
        num_epochs: int = 500,
        scenes_per_batch: int = 1,
        device: str = "cpu",
    ):
        self.base_directory = base_directory
        self.model_name = model_name
        self.scenes = scenes
        self.domain_radius = domain_radius
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.scenes_per_batch = scenes_per_batch
        self.device = device

        self.trained_scenes: Dict[str, Scene] = {}

        os.makedirs(self.base_directory, exist_ok=True)

    def _sample_scene(
        self,
        scene: SceneWithOperators,
        samples_per_scene: int,
        clamp_dist: float,
        outlier_pct: float = 0.05,
    ):
        """
        Robust DeepSDF-compatible sampling:
        - 50/50 positive / negative SDF
        - shell + volume sampling
        - bounded outliers
        - operator-parameter aware

        Returns:
            pts : (N, D)
            sdf : (N, 1)
        """

        device = torch.device("cpu")

        # ---------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------
        def sample_uniform_dirs(n):
            v = torch.randn(n, 3, device=device)
            return v / (v.norm(dim=1, keepdim=True) + 1e-12)

        def estimate_center(sdf_fn, probe_N=200_000):
            probe_pts = (torch.rand(probe_N, 3, device=device) * 2 - 1) * self.domain_radius
            sdf_vals = sdf_fn(probe_pts, None)
            sdf_vals = sdf_vals[:, 0] if sdf_vals.dim() == 2 else sdf_vals

            mask = torch.abs(sdf_vals) < max(0.3, clamp_dist * 3)
            if mask.sum() < 128:
                return torch.zeros(3, device=device)

            near_pts = probe_pts[mask]
            near_sdf = sdf_vals[mask]

            projected = near_pts - near_sdf.unsqueeze(1) * (
                near_pts / (near_pts.norm(dim=1, keepdim=True) + 1e-12)
            )
            return projected.median(dim=0).values

        # ---------------------------------------------------------
        # Scene center estimation
        # ---------------------------------------------------------
        any_sdf_fn, _ = next(iter(scene.values()))
        shape_center = estimate_center(any_sdf_fn)

        ops = list(scene.values())
        n_ops = len(ops)

        total_target = samples_per_scene // max(1, n_ops)
        target_pos = total_target // 2
        target_neg = total_target - target_pos

        pos_chunks, neg_chunks = [], []

        # =========================================================
        # Operator-wise sampling
        # =========================================================
        for sdf_fn, param_ranges in ops:
            n_params = len(param_ranges)

            if n_params > 0:
                low = torch.tensor([a for a, _ in param_ranges], device=device)
                high = torch.tensor([b for _, b in param_ranges], device=device)
            else:
                low = high = None

            # -----------------------------------------------------
            # Estimate surface radius
            # -----------------------------------------------------
            dirs = sample_uniform_dirs(2048)
            probes = shape_center.unsqueeze(0) + dirs * (self.domain_radius * 0.95)
            sd = sdf_fn(probes, None)
            sd = sd[:, 0] if sd.dim() == 2 else sd

            approx_r = (probes - shape_center.unsqueeze(0)).norm(dim=1) - sd
            R = float(torch.median(approx_r).clamp(min=1e-3))

            # -----------------------------------------------------
            # Sampling loop
            # -----------------------------------------------------
            op_pos, op_neg = [], []
            allowed_pos = int(outlier_pct * target_pos)
            allowed_neg = int(outlier_pct * target_neg)
            used_pos = used_neg = 0

            attempts = 0
            max_attempts = 20_000
            batch_size = 4096

            while (
                (len(op_pos) < target_pos or len(op_neg) < target_neg)
                and attempts < max_attempts
            ):
                attempts += 1

                pts = shape_center.unsqueeze(0) + (
                    torch.rand(batch_size, 3, device=device) * 2 - 1
                ) * max(R * 1.2, clamp_dist * 5)

                mask = (pts - shape_center.unsqueeze(0)).norm(dim=1) <= self.domain_radius
                pts = pts[mask]
                if pts.numel() == 0:
                    continue

                if n_params > 0:
                    rp = torch.rand(pts.shape[0], n_params, device=device)
                    params = low.unsqueeze(0) + rp * (high - low).unsqueeze(0)
                else:
                    params = None

                sdf_vals = sdf_fn(pts, params)
                sdf_vals = sdf_vals[:, 0] if sdf_vals.dim() == 2 else sdf_vals

                pts_np = pts.cpu().numpy()
                sdf_np = sdf_vals.cpu().numpy()

                if params is not None:
                    params_np = params.cpu().numpy()

                for i in range(len(sdf_np)):
                    v = sdf_np[i]
                    accept = abs(v) <= clamp_dist

                    if v >= 0:
                        if not accept and used_pos < allowed_pos:
                            accept = True
                            used_pos += 1
                        if accept and len(op_pos) < target_pos:
                            row = np.concatenate(
                                [pts_np[i], params_np[i] if params is not None else [], [v]]
                            )
                            op_pos.append(row)
                    else:
                        if not accept and used_neg < allowed_neg:
                            accept = True
                            used_neg += 1
                        if accept and len(op_neg) < target_neg:
                            row = np.concatenate(
                                [pts_np[i], params_np[i] if params is not None else [], [v]]
                            )
                            op_neg.append(row)

            # -----------------------------------------------------
            # Padding (never return empty)
            # -----------------------------------------------------
            sample_dim = 3 + n_params + 1
            fallback = np.zeros(sample_dim, dtype=np.float32)

            if len(op_pos) == 0:
                op_pos.append(fallback)
            if len(op_neg) == 0:
                op_neg.append(fallback)

            while len(op_pos) < target_pos:
                op_pos.append(op_pos[-1].copy())
            while len(op_neg) < target_neg:
                op_neg.append(op_neg[-1].copy())

            pos_chunks.append(np.vstack(op_pos))
            neg_chunks.append(np.vstack(op_neg))

        # =========================================================
        # Merge
        # =========================================================
        pos = np.vstack(pos_chunks)
        neg = np.vstack(neg_chunks)

        all_samples = np.vstack([pos, neg]).astype(np.float32)
        pts = all_samples[:, :-1]
        sdf = all_samples[:, -1:].reshape(-1, 1)

        return pts, sdf


    def train(self):
        print("[INFO] Sampling scenes")

        clamp_dist = 0.1
        samples_per_scene = 50_000

        scene_samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for idx, (scene_id, scene) in enumerate(self.scenes.items()):
            print(f"[SAMPLE] Scene '{scene_id}'")
            pts, sdf = self._sample_scene(
                scene,
                samples_per_scene,
                clamp_dist,
            )
            scene_samples[idx] = (pts, sdf)

        dataset = SceneSDFDataset(scene_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.scenes_per_batch,
            shuffle=True,
            drop_last=True,
        )

        # Geometry dimension = xyz + operator params
        example_pts, _ = next(iter(scene_samples.values()))
        input_dim = example_pts.shape[1]

        print(f"[INFO] Geometry input dimension: {input_dim}")

        model = DeepSDF(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=512,
            num_layers=8,
            latent_injection_layer=4,
        )

        trainer = DeepSDFTrainer(
            model=model,
            base_directory=self.base_directory,
            num_shapes=len(self.scenes),
            latent_dim=self.latent_dim,
            latent_sigma=0.01,
            lr_network=1e-4,
            lr_latent=1e-3,
            clamp_delta=clamp_dist,
            device=self.device,
        )

        print(f"[INFO] Training for {self.num_epochs} epochs")

        trainer.train(
            dataloader=loader,
            num_epochs=self.num_epochs,
            snapshot_every=10  # saves every 10 epochs
        )

        self.model = model
        self.trainer = trainer

        # Register trained scenes
        for idx, scene_id in enumerate(self.scenes.keys()):
            key = f"{self.model_name.lower()}_{scene_id}"
            latent = trainer.latents.weight[idx].detach().cpu()

            self.trained_scenes[key] = Scene(
                parent_model=self,
                scene_key=key,
                latent_vector=latent,
            )

        print(f"[INFO] Registered {len(self.trained_scenes)} trained scenes")


    def compute_sdf_from_latent(
        self,
        latent_vector: torch.Tensor,
        xyz: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        chunk: int = 50_000,
    ):
        self.model.eval()

        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)

        outputs = []

        with torch.no_grad():
            for i in range(0, xyz.shape[0], chunk):
                pts = xyz[i : i + chunk]

                if params is not None:
                    pts = torch.cat(
                        [pts, params.expand(pts.size(0), -1)], dim=1
                    )

                z = latent_vector.expand(pts.size(0), -1).to(pts.device)
                sdf = self.model(pts, z)
                outputs.append(sdf.squeeze(1))

        return torch.cat(outputs, dim=0)

    def get_scene(self, scene_key: str):
        return self.trained_scenes[scene_key]
    
    def visualize_a_shape(
        self,
        key: str,
        latent: torch.Tensor,
        grid_center=(0.0, 0.0, 0.0),
        grid_res=128,
        clamp_dist=0.1,
        param_values=None,
        save_suffix=None,
    ):
        """
        Visualize a trained scene from this Model instance.
        Uses self.trained_scenes and compute_trained_sdf internally.
        Fully CPU-compatible.
        """
        if param_values is None:
            param_values = [None]

        meshes = []

        # Build sampling grid
        xyz_points, x, y, z = self._build_grid(grid_center, grid_res)

        for idx, param_case in enumerate(param_values):
            pts = xyz_points.clone()
            if param_case is not None:
                param_tensor = torch.as_tensor(param_case, dtype=torch.float32).view(1, -1)
                pts = torch.cat([pts, param_tensor.repeat(pts.shape[0], 1)], dim=1)

            # Compute SDF using Scene's trained latent
    
            sdf = self.compute_sdf_from_latent(
                    latent_vector=latent,
                    xyz=pts,
                    params=param_tensor if param_case is not None else None,
                    chunk=50000,
                ).cpu().numpy()

            # Clip SDF and check for zero-crossing
            volume = np.clip(sdf.reshape(grid_res, grid_res, grid_res), -clamp_dist, clamp_dist)
            if not (volume.min() < 0 < volume.max()):
                print(f"[WARN] No zero-crossing found — skipping mesh for case {idx}")
                continue

            verts, faces, normals, _ = measure.marching_cubes(volume, level=0.0)
            scale = x[1] - x[0]
            verts = verts * scale + np.array([x[0], y[0], z[0]])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

            # Save mesh
            mesh_dir = os.path.join(self.base_directory, "Meshes")
            os.makedirs(mesh_dir, exist_ok=True)

            suffix_parts = []
            if param_case is not None:
                safe_param = "_".join([f"{v:.2f}".replace("-", "m").replace("+", "p")
                                    for v in np.atleast_1d(param_case)])
                suffix_parts.append(f"params{safe_param}")
            if save_suffix:
                suffix_parts.append(save_suffix)
            suffix_parts.append(f"case{idx:02d}")
            suffix_str = "_" + "_".join(suffix_parts)

            mesh_filename = f"{self.model_name.lower()}_{key}{suffix_str}_mesh.ply"
            mesh_path = os.path.join(mesh_dir, mesh_filename)
            mesh.export(mesh_path)
            print(f"[INFO] Saved mesh → {mesh_path}")

            meshes.append(mesh)

        return meshes


    def _build_grid(self, grid_center=(0.0, 0.0, 0.0), grid_res=128):
        """
        Internal helper: create a uniform sampling grid around a center.
        """
        bbox_half = 1.0
        x = np.linspace(grid_center[0] - bbox_half, grid_center[0] + bbox_half, grid_res)
        y = np.linspace(grid_center[1] - bbox_half, grid_center[1] + bbox_half, grid_res)
        z = np.linspace(grid_center[2] - bbox_half, grid_center[2] + bbox_half, grid_res)
        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        pts_flat = torch.from_numpy(grid.reshape(-1, 3)).float()
        return pts_flat, x, y, z



class Scene:
    def __init__(
        self,
        parent_model: Model,
        scene_key: str,
        latent_vector: torch.Tensor,
    ):
        self.parent_model = parent_model
        self.scene_key = scene_key
        self.latent_vector = latent_vector

        raw_id = "_".join(scene_key.split("_")[1:])
        self.sdf_ops = self.parent_model.scenes.get(raw_id)

        if self.sdf_ops is None:
            raise KeyError(f"Scene '{raw_id}' not found")

    def compute_trained_sdf(
        self,
        xyz: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        chunk: int = 50_000,
    ):
        return self.parent_model.compute_sdf_from_latent(
            latent_vector=self.latent_vector,
            xyz=xyz,
            params=params,
            chunk=chunk,
        )

    def get_latent_vector(self):
        return self.latent_vector



