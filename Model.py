from typing import Callable, Dict, Tuple, List, Optional
import os
from DeepSDFTrainer import DeepSDF, DeepSDFTrainer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import measure
import trimesh


SDFCallable = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
SceneWithOperators = Dict[int, Tuple[SDFCallable, List[Tuple[float, float]]]]
Scenes = Dict[str, SceneWithOperators]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def sample_uniform_dirs(n: int, device: torch.device) -> torch.Tensor:
    v = torch.randn(n, 3, device=device)
    return v / (v.norm(dim=1, keepdim=True) + 1e-12)

def estimate_center(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    probe_N: int,
    surface_thresh: float,
    init_range: float,
    fallback_center: torch.Tensor,
    device: torch.device,
):
    pts = (torch.rand(probe_N, 3, device=device) * 2 - 1) * init_range
    sdf_vals = sdf_fn(pts).view(-1)

    mask = torch.abs(sdf_vals) < surface_thresh
    if mask.sum() < 128:
        return fallback_center.clone()

    near_pts = pts[mask]
    near_sdf = sdf_vals[mask]

    dirs = near_pts / (near_pts.norm(dim=1, keepdim=True) + 1e-12)
    projected = near_pts - near_sdf.unsqueeze(1) * dirs

    return projected.median(dim=0).values




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
        training_clamp_dist: float|None = 0.1,
        sample_clamp_dist: float = 0.1,
        samples_per_scene: int = 5000,
        regularize_latent: bool = False,    latent_injection_layer: int = 4,
    
        soft_latent: bool = True,
    ):
        self.base_directory = base_directory
        self.regularize_latent = regularize_latent
        self.soft_latent = soft_latent
    
        self.model_name = model_name
        self.scenes = scenes
        self.domain_radius = domain_radius
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.scenes_per_batch = scenes_per_batch
        self.device = device
        self.training_clamp_dist = training_clamp_dist
        self.sample_clamp_dist = sample_clamp_dist
        self.samples_per_scene = samples_per_scene
        self.latent_injection_layer = latent_injection_layer

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
        # Scene center estimation
        # ---------------------------------------------------------
        any_sdf_fn, _ = next(iter(scene.values()))

        def sdf_eval(xyz: torch.Tensor) -> torch.Tensor:
            sdf = any_sdf_fn(xyz, None)
            return sdf[:, 0] if sdf.dim() == 2 else sdf

        shape_center = estimate_center(
            sdf_fn=sdf_eval,
            probe_N=200_000,
            surface_thresh=0.3,
            init_range=self.domain_radius,
            fallback_center=torch.zeros(3, device=device),
            device=device,
        )


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
            dirs = sample_uniform_dirs(2048, device=device)
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
            max_attempts = 50_000
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

  
        

        scene_samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for idx, (scene_id, scene) in enumerate(self.scenes.items()):
            print(f"[SAMPLE] Scene '{scene_id}'")
            pts, sdf = self._sample_scene(
                scene,
                self.samples_per_scene,
                clamp_dist=self.sample_clamp_dist,
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
            latent_injection_layer=self.latent_injection_layer,
            soft_latent=self.soft_latent,
        )

        trainer = DeepSDFTrainer(
            model=model,
            base_directory=self.base_directory,
            num_shapes=len(self.scenes),
            latent_dim=self.latent_dim,
            clamp_delta= self.training_clamp_dist,
            device=self.device,
            regularize_latent=self.regularize_latent,
        )

        print(f"[INFO] Training for {self.num_epochs} epochs")

        trainer.train(
            dataloader=loader,
            epochs=self.num_epochs,
            snapshot_every=100  # saves every 100 epochs
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
        grid_res=128,
        clamp_dist=0.1,
        param_values=None,
        save_suffix=None,
    ):
        if param_values is None:
            param_values = [None]

        meshes = []

        latent_vector = latent.view(1, -1).float()
        decoder = self.model
        device = next(decoder.parameters()).device

        example_scene = next(iter(self.scenes.values()))
        _, param_ranges = next(iter(example_scene.values()))
        num_params = len(param_ranges)

        xyz_points, x, y, z = self.build_dynamic_sampling_grid(
            latent_vector=latent_vector,
            grid_res=grid_res,
            device=device,
        )

        for idx, param_case in enumerate(param_values):
            pts = xyz_points.clone()

            param_tensor = None
            if param_case is not None:
                param_tensor = torch.tensor(param_case).view(1, -1)
                pts = torch.cat([pts, param_tensor.repeat(len(pts), 1)], dim=1)

            sdf = self.compute_sdf_from_latent(
                latent_vector=latent_vector,
                xyz=pts,
                params=param_tensor,
            ).cpu().numpy()

            volume = np.clip(sdf.reshape(grid_res, grid_res, grid_res), -clamp_dist, clamp_dist)
            if not (volume.min() < 0 < volume.max()):
                continue

            verts, faces, normals, _ = measure.marching_cubes(volume, level=0.0)
            scale = x[1] - x[0]
            verts = verts * scale + np.array([x[0], y[0], z[0]])

            mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals)

            mesh_dir = os.path.join(self.base_directory, "Meshes")
            os.makedirs(mesh_dir, exist_ok=True)

            suffix = f"_case{idx:02d}"
            if save_suffix:
                suffix += f"_{save_suffix}"

            mesh_path = os.path.join(
                mesh_dir, f"{self.model_name.lower()}_{key}{suffix}_mesh.ply"
            )

            mesh.export(mesh_path)
            meshes.append(mesh)

        return meshes
    
    def build_dynamic_sampling_grid(
        self,
        latent_vector: torch.Tensor,
        grid_res: int,
        init_range: float = 3.0,
        probe_N: int = 500_000,
        surface_thresh: float = 0.3,
        n_surface_probes: int = 2048,
        bbox_margin_ratio: float = 0.12,
        fallback_center=(0.0, 0.0, 0.0),
        device: Optional[torch.device] = None,
    ):
        """
        Builds a sampling grid tightly enclosing the zero level set
        of a learned DeepSDF shape.

        This function:
        - uses ONLY sdf_fn(xyz) → sdf
        - takes the latent vector explicitly
        - performs center + radius estimation once
        """

        if device is None:
            device = latent_vector.device

        latent_vector = latent_vector.to(device)

        fallback_center = torch.tensor(
            fallback_center, dtype=torch.float32, device=device
        )

        # ------------------------------------------------------------
        # Center estimation
        # ------------------------------------------------------------
        pts = (torch.rand(probe_N, 3, device=device) * 2 - 1) * init_range
        sdf_vals = self.compute_sdf_from_latent(latent_vector, pts).view(-1)

        near_mask = torch.abs(sdf_vals) < surface_thresh
        if near_mask.sum() < 128:
            center = fallback_center.clone()
        else:
            near_pts = pts[near_mask]
            near_sdf = sdf_vals[near_mask]

            dirs = near_pts / (near_pts.norm(dim=1, keepdim=True) + 1e-12)
            projected = near_pts - near_sdf.unsqueeze(1) * dirs
            center = projected.median(dim=0).values

        # ------------------------------------------------------------
        # Radius estimation
        # ------------------------------------------------------------
        dirs = sample_uniform_dirs(n_surface_probes, device)
        probes = center.unsqueeze(0) + dirs * (init_range * 0.95)

        sd = self.compute_sdf_from_latent(latent_vector, probes)
        radii = (probes - center).norm(dim=1) - sd

        radius = torch.median(radii).clamp(min=1e-3).item()
        margin = radius * bbox_margin_ratio

        # ------------------------------------------------------------
        # Grid construction
        # ------------------------------------------------------------
        lo = center.cpu().numpy() - (radius + margin)
        hi = center.cpu().numpy() + (radius + margin)

        hi = np.maximum(hi, lo + 1e-6)

        x = np.linspace(lo[0], hi[0], grid_res)
        y = np.linspace(lo[1], hi[1], grid_res)
        z = np.linspace(lo[2], hi[2], grid_res)

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



