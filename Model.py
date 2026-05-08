from typing import Callable, Dict, Tuple, List, Optional
import os
from DeepSDFTrainer import DeepSDF, DeepSDFTrainer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import measure
import pyrender
from PIL import Image, ImageEnhance, ImageFilter
import trimesh
SDFCallable = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
SceneWithOperators = Dict[int, Tuple[SDFCallable, List[Tuple[float, float]]]]
Scenes = Dict[str, SceneWithOperators]


# Helpers
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
        return fallback_center.clone().to(device)

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
        scenes_per_batch: int = 16,
        training_clamp_dist: float|None = 0.1,
        sample_clamp_dist: float = 0.1,
        samples_per_scene: int = 50000,
        regularize_latent: bool = False,
        skip_layer: int | None = 4,
        soft_latent: bool = True,
        weight_norm: bool = False,
        train_until_convergence: bool = False,
        stochastic_distribution: bool = True, 
        sigma0=1e-4,
        lr_net=5e-4, 
        lr_latent=1e-3, 
        window = 50,
        #c= 2, 
        patience = 1,
        min_delta=0.05
    ):
        self.base_directory = base_directory
        self.regularize_latent = regularize_latent
        self.soft_latent = soft_latent
        self.weight_norm = weight_norm
        self.train_until_convergence = train_until_convergence
        self.model_name = model_name
        self.scenes = scenes
        self.domain_radius = domain_radius
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.scenes_per_batch = scenes_per_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.training_clamp_dist = training_clamp_dist
        self.sample_clamp_dist = sample_clamp_dist
        self.samples_per_scene = samples_per_scene
        self.skip_layer = skip_layer
        self.stochastic_distribution = stochastic_distribution

        self.trained_scenes: Dict[str, Scene] = {}
        self.sigma0 = sigma0
        self.lr_net = lr_net
        self.lr_latent = lr_latent
        self.window = window
        #self.c = c
        self.patience= patience
        self.min_delta = min_delta

        os.makedirs(self.base_directory, exist_ok=True)

    @classmethod
    def from_snapshot(
        cls,
        snapshot_path: str,
        *,
        base_directory: str,
        model_name: str,
        latent_injection_layer: int | None = None,
        soft_latent: bool = True,
        weight_norm: bool = False,
    ):
        """
        Load a trained DeepSDF Model from a snapshot .pth file for inference.
        Fully reconstructs the decoder to match checkpoint shapes exactly.

        Don't use unless you're in a pinch this thing is a total mess
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(snapshot_path, map_location=device)
        state_dict = ckpt["model_state_dict"]

        # Determine decoder parameters from checkpoint
        layer_keys = sorted([k for k in state_dict.keys() if k.startswith("layers.") and k.endswith("weight")])
        num_layers = len(layer_keys)
        first_layer_weight = state_dict[layer_keys[0]]
        hidden_dim = first_layer_weight.shape[0]
        input_dim_with_latent = first_layer_weight.shape[1]  # includes latent concat

        latent_state = ckpt["latent_codes"]
        latent_dim = latent_state["weight"].shape[1]

        # Infer latent injection layer if not provided
        if latent_injection_layer is None:
            latent_injection_layer = None
            for i, k in enumerate(layer_keys[1:], start=1):
                w = state_dict[k]
                if w.shape[1] == hidden_dim + latent_dim:
                    latent_injection_layer = i
                    break

        # Compute actual input_dim to pass to DeepSDF (subtract latent_dim)
        input_dim = input_dim_with_latent - latent_dim

        print(f"[INFO] Inferred decoder: input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, latent_dim={latent_dim}, latent_injection_layer={latent_injection_layer}")

        # Rebuild decoder exactly
        decoder = DeepSDF(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_layer=latent_injection_layer,
            soft_latent=soft_latent,
            weight_norm= weight_norm,
        ).to(device)

        decoder.load_state_dict(state_dict)
        decoder.eval()

        # Instantiate empty model shell
        model_obj = cls(
            base_directory=base_directory,
            model_name=model_name,
            scenes={},  # inference-only
            latent_dim=latent_dim,
            skip_layer=latent_injection_layer,
            soft_latent=soft_latent,
        
            num_epochs=0,
        )
        model_obj.model = decoder
        model_obj.trainer = None


        # Load latent embedding
        latent_embedding = torch.nn.Embedding(
            num_embeddings=latent_state["weight"].shape[0],
            embedding_dim=latent_state["weight"].shape[1],
        )
        latent_embedding.load_state_dict(latent_state)
        latent_embedding = latent_embedding.to(device)

        # Register latents
        model_obj.trained_scenes = {}
        for idx in range(latent_embedding.weight.shape[0]):
            key = f"{model_name.lower()}_latent_{idx}"
            model_obj.trained_scenes[key] = latent_embedding.weight[idx].detach().cpu()

        print(
            f"[INFO] Loaded model from snapshot '{snapshot_path}' "
            f"(epoch {ckpt.get('epoch', 'unknown')}) "
            f"with {len(model_obj.trained_scenes)} latent vectors"
        )

        return model_obj



    def _sample_scene(
        self,
        scene: SceneWithOperators,
        samples_per_scene: int,
        clamp_dist: float,
        outlier_pct: float = 0.05,
    ):
        """
        Samples a scene by querying its SDF operators.
        - 50/50 positive / negative SDF
        - shell + volume sampling
        - bounded outliers
        - operator-parameter aware

        Returns:
            pts : (N, D)
            sdf : (N, 1)
        """

        device = self.device
   
        # Scene center estimation
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

        # Operator-wise sampling
        for sdf_fn, param_ranges in ops:
            n_params = len(param_ranges)

            if n_params > 0:
                low = torch.tensor([a for a, _ in param_ranges], device=device, dtype=torch.float32)
                high = torch.tensor([b for _, b in param_ranges], device=device, dtype=torch.float32)
            else:
                low = high = None

            # Estimate surface radius
            dirs = sample_uniform_dirs(2048, device=device)
            probes = shape_center.unsqueeze(0) + dirs * (self.domain_radius * 0.95)
            sd = sdf_fn(probes, None)
            sd = sd[:, 0] if sd.dim() == 2 else sd

            approx_r = (probes - shape_center.unsqueeze(0)).norm(dim=1) - sd
            R = float(torch.median(approx_r).clamp(min=1e-3))

            # Sampling loop
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
    
    def _sample_scene_over_grid(
        self,
        scene: SceneWithOperators,
        grid_pts: torch.Tensor,
    ):
        """
        Samples a scene by evaluating the SDF on a provided grid.

        Unlike _sample_scene():
            - NO positive/negative balancing
            - NO clamp filtering
            - NO adaptive sampling
            - simply evaluates SDF on every grid point

        Parameters
        ----------
        scene : SceneWithOperators
            Scene definition (SDF operators)

        grid_pts : torch.Tensor
            (N,3) grid of xyz coordinates

        Returns
        -------
        pts : np.ndarray
            (N, D) where D = xyz + params

        sdf : np.ndarray
            (N,1)
        """

        device = self.device
        grid_pts = grid_pts.to(device)

        all_pts = []
        all_sdf = []

        for sdf_fn, param_ranges in scene.values():

            n_params = len(param_ranges)

            # ----------------------------------------
            # Parameter handling
            # ----------------------------------------
            if n_params > 0:

                low = torch.tensor(
                    [a for a, _ in param_ranges],
                    device=device,
                    dtype=torch.float32,
                )

                high = torch.tensor(
                    [b for _, b in param_ranges],
                    device=device,
                    dtype=torch.float32,
                )

                rp = torch.rand(grid_pts.shape[0], n_params, device=device)
                params = low.unsqueeze(0) + rp * (high - low).unsqueeze(0)

            else:
                params = None

            # ----------------------------------------
            # Evaluate SDF
            # ----------------------------------------
            sdf_vals = sdf_fn(grid_pts, params)

            if sdf_vals.dim() == 2:
                sdf_vals = sdf_vals[:, 0]

            # ----------------------------------------
            # Convert to numpy
            # ----------------------------------------
            pts_np = grid_pts.cpu().numpy()
            sdf_np = sdf_vals.detach().cpu().numpy().reshape(-1, 1)

            if params is not None:
                params_np = params.cpu().numpy()
                pts_np = np.concatenate([pts_np, params_np], axis=1)

            all_pts.append(pts_np)
            all_sdf.append(sdf_np)

        pts = np.vstack(all_pts).astype(np.float32)
        sdf = np.vstack(all_sdf).astype(np.float32)

        return pts, sdf


    def train(self, grid= None):
        stochastic = self.stochastic_distribution
        print("[INFO] Sampling scenes")

        scene_samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if grid == None: 
            for idx, (scene_id, scene) in enumerate(self.scenes.items()):
                print(f"[SAMPLE] Scene '{scene_id}'")
                pts, sdf = self._sample_scene(
                    scene,
                    self.samples_per_scene,
                    clamp_dist=self.sample_clamp_dist,
                )
                scene_samples[idx] = (pts, sdf)
        else:
            for idx, (scene_id, scene) in enumerate(self.scenes.items()):
                print(f"[SAMPLE] Scene '{scene_id}'")
                
                pts, sdf = self._sample_scene_over_grid(scene = scene, grid_pts=grid)
    
                scene_samples[idx] = (pts, sdf)

        dataset = SceneSDFDataset(scene_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.scenes_per_batch,
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
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
            skip_layer=self.skip_layer,
            soft_latent=self.soft_latent,
            weight_norm= self.weight_norm,
        ).to(self.device)

        trainer = DeepSDFTrainer(
            model=model,
            base_directory=self.base_directory,
            num_shapes=len(self.scenes),
            latent_dim=self.latent_dim,
            clamp_delta= self.training_clamp_dist,
            regularize_latent=self.regularize_latent,
            sigma0=self.sigma0,
            lr_net=self.lr_net,
            lr_latent=self.lr_latent,
            window=self.window,
            #c=self.c,
            patience=self.patience,
            min_delta=self.min_delta
        )

        print(f"[INFO] Training for {self.num_epochs} epochs")

        trainer.train(
            dataloader=loader,
            epochs=self.num_epochs,
            snapshot_every=1000,
            stochastic_distribution=stochastic,
            train_until_convergence=self.train_until_convergence

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
        chunk: int = 50000,
    ):
        self.model.eval()

        device = self.device

        latent_vector = latent_vector.to(device)

        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)

        xyz = xyz.to(device)

        if params is not None:
            params = params.to(device)

        outputs = []

        with torch.no_grad():
            for i in range(0, xyz.shape[0], chunk):

                pts = xyz[i:i+chunk]

                if params is not None:
                    p = params.expand(pts.size(0), -1)
                    pts = torch.cat([pts, p], dim=1)

                z = latent_vector.expand(pts.size(0), -1)

                sdf = self.model(pts, z)

                outputs.append(sdf.squeeze(1))

        return torch.cat(outputs, dim=0)

    def get_scene(self, scene_key: str):
        return self.trained_scenes[scene_key]
    
    
    
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
        pts = pts.to(device)
        sdf_vals = self.compute_sdf_from_latent(latent_vector, pts).view(-1)
        sdf_vals = sdf_vals.to(device)

        near_mask = torch.abs(sdf_vals) < surface_thresh
        sdf_vals = sdf_vals.to(device)
        if near_mask.sum() < 128:
            center = fallback_center.clone()
        else:
            near_pts = pts[near_mask]
            near_pts = near_pts.to(device)
            near_sdf = sdf_vals[near_mask]
            near_sdf = near_sdf.to(device)

            dirs = near_pts / (near_pts.norm(dim=1, keepdim=True) + 1e-12)
            projected = near_pts - near_sdf.unsqueeze(1) * dirs
            center = projected.median(dim=0).values

        # Radius estimation
        dirs = sample_uniform_dirs(n_surface_probes, device)
        probes = center.unsqueeze(0) + dirs * (init_range * 0.95)

        sd = self.compute_sdf_from_latent(latent_vector, probes)
        sd=sd.to(device)
        radii = (probes - center).norm(dim=1) - sd

        radius = torch.median(radii).clamp(min=1e-3).item()
        margin = radius * bbox_margin_ratio

        # Grid construction
        lo = center.cpu().numpy() - (radius + margin)
        hi = center.cpu().numpy() + (radius + margin)

        hi = np.maximum(hi, lo + 1e-6)

        x = np.linspace(lo[0], hi[0], grid_res)
        y = np.linspace(lo[1], hi[1], grid_res)
        z = np.linspace(lo[2], hi[2], grid_res)

        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        pts_flat = torch.from_numpy(grid.reshape(-1, 3)).float().to(device)


        return pts_flat, x, y, z

    # SDF → voxel occupancy
    def sdf_voxel(self, latent,grid_pts,GRID_RES):

        with torch.no_grad():

            sdf = self.compute_sdf_from_latent(
                latent,
                grid_pts,
                chunk=50000
            )

        occ = (sdf < 0).cpu().numpy()

        return occ.reshape(GRID_RES, GRID_RES, GRID_RES)



    # SDF → mesh reconstruction
    def reconstruct_mesh(self, latent, name, grid_pts,GRID_RES):

        with torch.no_grad():

            sdf = self.compute_sdf_from_latent(
                latent,
                grid_pts,
                chunk=50000
            )

        sdf = sdf.cpu().numpy().reshape(GRID_RES,GRID_RES,GRID_RES)

        verts, faces, _, _ = measure.marching_cubes(
            sdf,
            level=0.0,
            spacing=(2/(GRID_RES-1), 2/(GRID_RES-1), 2/(GRID_RES-1))
        )

        verts -= 1.0

        mesh = trimesh.Trimesh(verts, faces)

        return mesh

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

        raw_id = scene_key

        if raw_id not in self.parent_model.scenes:
            raw_id = scene_key.split(self.parent_model.model_name.lower() + "_")[-1]

        self.sdf_ops = self.parent_model.scenes.get(raw_id)

        if self.sdf_ops is None:
            raise KeyError(f"Scene '{raw_id}' not found")


    def compute_trained_sdf(
        self,
        xyz: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        chunk: int = 50000,
    ):
        
        return self.parent_model.compute_sdf_from_latent(
            latent_vector=self.latent_vector.to(self.parent_model.device),
            xyz=xyz,
            params=params,
            chunk=chunk,
        )
    
    def get_latent_vector(self):
        return self.latent_vector




def render_mesh_isometric(mesh: trimesh.Trimesh, resolution=(500, 500)):
    """
    Render a trimesh mesh in an isometric view using pyrender offscreen.
    
    Parameters:
        mesh: trimesh.Trimesh
            The mesh to render.
        resolution: tuple(int, int)
            Image resolution (width, height).
    
    Returns:
        numpy.ndarray
            Rendered image as an array of shape (H, W, 3) in RGB.
    """
    # Create a pyrender scene
    scene = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[0.5, 0.5, 0.5])

    # Convert trimesh mesh to pyrender mesh and add to scene
    mesh_pyr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_pyr)

    # Camera setup: isometric-like view
    # We'll compute a distance that frames the mesh
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    size = np.linalg.norm(bounds[1] - bounds[0])
    if size < 1e-6:
        size = 1.0

    # Simple rotation matrix for an isometric-ish view
    # Pitch down 45 deg, yaw 0 deg, roll 15 deg
    pitch, yaw, roll = np.radians([45, 0, 15])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Camera position
    cam_distance = size * 2.0
    cam_position = center + R @ np.array([0, 0, cam_distance])
    cam_target = center

    # Compute look-at matrix
    def look_at(cam_pos, target):
        forward = (target - cam_pos)
        forward /= np.linalg.norm(forward)
        right = np.cross([0, 0, 1], forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        mat = np.eye(4)
        mat[:3, 0] = right
        mat[:3, 1] = up
        mat[:3, 2] = forward
        mat[:3, 3] = cam_pos
        return mat

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(camera, pose=look_at(cam_position, cam_target))

    # Add a simple directional light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=np.eye(4))

    # Offscreen renderer
    r = pyrender.OffscreenRenderer(*resolution)
    color, depth = r.render(scene)
    r.delete()

    return color  # RGB image as numpy array



def render_mesh_isometric_pil(mesh: trimesh.Trimesh,
                         resolution=(500,500),
                         pitch_deg=-45,
                         yaw_deg=15,
                         distance_factor=1.2,
                         contrast_factor=3,
                         sharpen=True) -> Image.Image:
    """
    Render mesh centered in the image. Mesh rotation defines the view (pitch/yaw),
    while the camera stays on +Z to avoid projection offsets.
    """
    mesh_copy = mesh.copy()
    center = mesh_copy.bounds.mean(axis=0)
    mesh_copy.apply_translation(-center)

    # Rotate mesh for isometric view
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    R_pitch = trimesh.transformations.rotation_matrix(pitch, [1,0,0])
    R_yaw = trimesh.transformations.rotation_matrix(yaw, [0,1,0])
    mesh_copy.apply_transform(R_yaw @ R_pitch)

    # Scene
    scene = pyrender.Scene(
    bg_color=[220, 220, 220],
    ambient_light=[0.03, 0.03, 0.03]  # NOT 0.2
    )

    blue_material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.18, 0.35, 0.85, 1.0],
    # deep but not saturated blue
    metallicFactor=0.1,
    roughnessFactor=0.4
    )

    scene.add(
        pyrender.Mesh.from_trimesh(
            mesh_copy,
            material=blue_material,
            smooth=True
        )
    )


    # Camera: fixed on +Z looking at origin
    diag = np.linalg.norm(mesh_copy.extents)
    cam_distance = diag * distance_factor
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    cam_pose = np.eye(4)
    cam_pose[:3,3] = [0,0,cam_distance]  # +Z
    scene.add(camera, pose=cam_pose)

    # Lights: relative to camera
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.0),
              pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
              pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
              pose=cam_pose)

    # Render
    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    r.delete()

    img = Image.fromarray(color[:, :, :3])

    # Boost contrast
    if contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Sharpen
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=3))

    return img
