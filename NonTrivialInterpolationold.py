from PIL import Image, ImageDraw, ImageFont
import os
import torch
import matplotlib.pyplot as plt
import trimesh
import imageio
import numpy as np

import Model
from VisualizeAnalyticSDF import visualize_analytic_sdf

def main():


    # ======================================================
    # Experiment Setup
    # ======================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(SCRIPT_DIR)
    EXPERIMENT_NAME = "NonTrivialInterpolation"
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_ROOT, "Meshes"), exist_ok=True)

    mesh_output_dir = os.path.join(EXPERIMENT_ROOT, "Meshes")


    # ======================================================
    # Analytic SDFs
    # ======================================================
    R = 0.55

    def analytic_torus_sdf(xyz, params=None):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        R_major, r_minor = 0.38, 0.17
        q = torch.sqrt(x**2 + y**2) - R_major
        return (torch.sqrt(q**2 + z**2) - r_minor).unsqueeze(1)

    def wavey_rounded_box_sdf(xyz, params=None):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        hx, hy, hz, r = 0.48, 0.48, 0.48, 0.10
        q = torch.stack([
            torch.abs(x) - hx,
            torch.abs(y) - hy,
            torch.abs(z) - hz
        ], dim=1)
        outside = torch.clamp(q, min=0.0)
        inside = torch.clamp(torch.max(q, dim=1).values, max=0.0)
        box = torch.linalg.norm(outside, dim=1) + inside - r
        bubble = 0.08 * torch.sin(3*x) * torch.sin(2.5*y) * torch.sin(2*z)
        return (box + bubble).unsqueeze(1)

    def dented_sphere_sdf(xyz, params=None):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        base = torch.sqrt(x**2 + y**2 + z**2) - R
        theta = torch.atan2(y, x)
        phi = torch.atan2(z, torch.sqrt(x**2 + y**2))
        dents = 0.10 * torch.cos(4*theta) * torch.cos(3*phi)
        return (base + dents).unsqueeze(1)


    scenes = {
        "torus": {0: (analytic_torus_sdf, [])},
        "wavey_rounded_box": {0: (wavey_rounded_box_sdf, [])},
        "dented_sphere": {0: (dented_sphere_sdf, [])},
    }

    visualize_analytic_sdf(analytic_torus_sdf, "torus", EXPERIMENT_ROOT, grid_res=256)
    visualize_analytic_sdf(wavey_rounded_box_sdf, "wavey_rounded_box", EXPERIMENT_ROOT, grid_res=256)
    visualize_analytic_sdf(dented_sphere_sdf, "dented_sphere", EXPERIMENT_ROOT, grid_res=256)


    # ======================================================
    # Train DeepSDF
    # ======================================================
    model = Model.Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="NonTrivialInterpolationtest",
        scenes=scenes,
        latent_dim=3,
        training_clamp_dist=None,
        sample_clamp_dist=1,
        num_epochs=2000,
        domain_radius=1.0,
        regularize_latent=True,
        soft_latent=False,
        samples_per_scene=5000,
        skip_layer=4
        
    )
    model.train()


    # ======================================================
    # Voxelization
    # ======================================================
    def mesh_to_fixed_voxel(mesh, grid_res=64, domain_radius=1.0):
        xs = np.linspace(-domain_radius, domain_radius, grid_res)
        ys = np.linspace(-domain_radius, domain_radius, grid_res)
        zs = np.linspace(-domain_radius, domain_radius, grid_res)
        pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
        occ = mesh.contains(pts).reshape(grid_res, grid_res, grid_res)
        return occ.astype(np.bool_)


    # ======================================================
    # Canonical meshes and latents
    # ======================================================
    canonical_voxels = {}
    latent_dict = {}

    for k, scene in model.trained_scenes.items():
        short = "_".join(k.split("_")[1:])
        latent_dict[short] = scene.latent_vector

    for k, z in latent_dict.items():
        mesh = model.visualize_a_shape(z, grid_res=128, clamp_dist=0.1)[0]
        canonical_voxels[k] = mesh_to_fixed_voxel(mesh)


    # ======================================================
    # Invariant leakage metric
    # ======================================================
    def topo_sim_voxel(A, B):
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        return inter / union if union > 0 else 0.0

    def invariant_leakage(z, intended):
        meshes = model.visualize_a_shape(z, grid_res=128, clamp_dist=0.1)
        if not meshes:
            return 0.0, None

        M = mesh_to_fixed_voxel(meshes[0])
        sims = {k: topo_sim_voxel(M, v) for k, v in canonical_voxels.items()}

        unintended = [k for k in sims if k not in intended]

        # ---- CRITICAL EDGE CASE ----
        if len(unintended) == 0:
            return 0.0, meshes[0]

        intended_mean = np.mean([sims[k] for k in intended])
        unintended_max = max(sims[k] for k in unintended)

        return max(0.0, unintended_max - intended_mean), meshes[0]


    # ======================================================
    # Interpolations
    # ======================================================
    z_torus = latent_dict["torus"]
    z_box = latent_dict["wavey_rounded_box"]
    z_sphere = latent_dict["dented_sphere"]

    d_box = z_box - z_torus
    d_sphere = z_sphere - z_torus

    segments = [
        {"type": "pair", "anchors": ["torus", "wavey_rounded_box"], "label": "Torus→Wavey Box"},
        {"type": "pair", "anchors": ["torus", "dented_sphere"], "label": "Torus→Dented Sphere"},
        {"type": "pair", "anchors": ["dented_sphere", "wavey_rounded_box"], "label": "Dented Sphere→Wavey Box"},
        {"type": "multi", "label": "Superposition"},
    ]

    ts = torch.linspace(0, 1, 30)
    gif_frames = []
    leakage_log = {}


    # ======================================================
    # Main loop
    # ======================================================
    for seg_idx, seg in enumerate(segments):
        leakage_log[seg_idx] = []

        for i, t in enumerate(ts):
            if seg["type"] == "pair":
                a, b = seg["anchors"]
                z = (1 - t) * latent_dict[a] + t * latent_dict[b]
                intended = {a, b}
            else:
                z = z_torus + t * d_box + t * d_sphere
                intended = {"torus", "wavey_rounded_box", "dented_sphere"}

            leak, mesh = invariant_leakage(z, intended)
            leakage_log[seg_idx].append(leak)

            if mesh is None:
                continue

            mesh.export(os.path.join(mesh_output_dir, f"seg{seg_idx}_{i:03d}.ply"))

            img = Model.render_mesh_isometric_pil(mesh)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), seg["label"], fill=(0, 0, 0), font=font)
            draw.text((10, 35), f"t={float(t):.3f}", fill=(0, 0, 0), font=font)
            draw.text((10, 60), f"Leak={leak:.4f}", fill=(0, 0, 0), font=font)

            frame_path = os.path.join(EXPERIMENT_ROOT, "plots", f"frame_{seg_idx}_{i:03d}.png")
            img.save(frame_path)
            gif_frames.append(frame_path)


    # ======================================================
    # Leakage plot
    # ======================================================
    plt.figure(figsize=(8, 5))
    for seg_idx, curve in leakage_log.items():
        plt.plot(ts.numpy(), curve, label=segments[seg_idx]["label"])

    plt.xlabel("Interpolation parameter t")
    plt.ylabel("Shape Leakage")
    plt.title("Shape leakage during linear latent interpolation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(EXPERIMENT_ROOT, "plots", "leakage_curves.png"))
    plt.close()
    


    # ======================================================
    # GIF
    # ======================================================
    imageio.mimsave(
        os.path.join(EXPERIMENT_ROOT, "plots", "interpolation.gif"),
        [imageio.imread(p) for p in gif_frames],
        duration=0.08
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()