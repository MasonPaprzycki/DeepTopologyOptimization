import os
import numpy as np
import torch
import matplotlib
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from DeepSDFStruct.sdf_primitives import SphereSDF
from Model import Model
import matplotlib.colors as mcolors
import multiprocessing as mp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#this experiment produces dual spheres if sampling is clamped and has
#full numerical uniformity
#
# If sampling is moderately variant but follows ideal deepSDF conditons (theoretically) 
#this experiment will produce a cylinder with a bubble that flows through it during interpolation 
#

def main(): 
    # ======================================================
    # Experiment Setup
    # ======================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(SCRIPT_DIR)

    EXPERIMENT_NAME = "SlidingSphere2D"
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

    FRAME_DIR = os.path.join(EXPERIMENT_ROOT, "frames_latents")
    PLOT_DIR = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR = os.path.join(EXPERIMENT_ROOT, "Meshes")

    for d in [EXPERIMENT_ROOT,
              os.path.join(EXPERIMENT_ROOT, "frames"),
              FRAME_DIR,
              PLOT_DIR,
              MESH_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")

    # ======================================================
    # Generate scenes
    # ======================================================
    def make_sphere_scene(cx: float):
        def sdf_fn(xyz, params=None):
            return SphereSDF(
                center=torch.tensor([cx, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device),
                radius=0.5
            )._compute(xyz)
        return sdf_fn

    num_scenes = 25
    x_positions = np.linspace(-0.8, 0.8, num_scenes)

    scenes = {
        f"sphere_{cx:.2f}": {0: (make_sphere_scene(cx), [])}
        for cx in x_positions
    }

    print(f"[INFO] Created {num_scenes} scenes")

    # ======================================================
    # Train Model
    # ======================================================
    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="SlidingSphereModel2D",
        scenes=scenes,
        latent_dim=1,
        num_epochs=1,
    )

    print("[INFO] Training model...")
    model.train()
    print("[INFO] Training complete.")

    # ======================================================
    # Sort trained scenes
    # ======================================================
    def extract_x(k: str):
        return float(k.split("_")[-1])

    sorted_keys = sorted(model.trained_scenes.keys(), key=extract_x)
    sorted_scenes = [model.trained_scenes[k] for k in sorted_keys]

    latents = torch.stack([s.get_latent_vector() for s in sorted_scenes]).float()
    latent_min, latent_max = latents[0], latents[-1]

    print("[INFO] Latent min/max:", latent_min.item(), latent_max.item())

    # ======================================================
    # Build 2D grid
    # ======================================================
    grid_res = 256
    xv = np.linspace(-1.8, 1.8, grid_res)
    yv = np.linspace(-1.8, 1.8, grid_res)
    xx, yy = np.meshgrid(xv, yv)

    xyz_np = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    xyz = torch.tensor(xyz_np, dtype=torch.float32).reshape(-1, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = xyz.to(device)

    # ======================================================
    # SDF evaluation
    # ======================================================
    def eval_sdf(latent_vec: torch.Tensor):
        sdf = model.compute_sdf_from_latent(latent_vector=latent_vec, xyz=xyz, params=None)
        if sdf.dim() == 2:
            sdf = sdf[:, 0]
        return sdf.cpu().numpy().reshape(grid_res, grid_res)

    # ======================================================
    # Custom colormap
    # ======================================================
    cdict = {
        "red":   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "green": [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
        "blue":  [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
    }

    custom_cmap = mcolors.LinearSegmentedColormap("sdf_custom", cdict)
    norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)

    # ======================================================
    # Latent-space sweep
    # ======================================================
    interp_steps = 100
    latent_values = torch.linspace(0.0, 1.0, interp_steps).to(device)

    print("[INFO] Generating latent-sweep frames...")
    frames = []

    for i, t in enumerate(latent_values):
        latent_vec = (1 - t) * latent_min + t * latent_max
        sdf_img = eval_sdf(latent_vec)

        plt.figure(figsize=(5, 5))
        plt.imshow(sdf_img, extent=(-1.8,1.8, -1.8,1.8),
                   cmap=custom_cmap, norm=norm, origin="lower")
        plt.colorbar(label="SDF")
        plt.title(f"t = {t.item():.2f}")
        plt.xlabel("x")
        plt.ylabel("y")

        frame_path = os.path.join(FRAME_DIR, f"latent_{i:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close()

        frames.append(imageio.imread(frame_path))

    print("[INFO] Frames generated.")

    gif_path = os.path.join(EXPERIMENT_ROOT, "latent_space_sweep.gif")
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"[INFO] Saved: {gif_path}")

    # ======================================================
    # Collect 1D latent codes
    # ======================================================
    print("[DEBUG] Scene keys:", sorted_keys)

    latents_np = latents.cpu().numpy()

    plt.figure(figsize=(6, 3))
    plt.scatter(latents_np[:, 0], np.zeros_like(latents_np[:,0]),
                color='royalblue', s=80)

    for i, key in enumerate(sorted_keys):
        plt.text(latents_np[i,0] + 0.01, 0.0, key.split("_")[-1], fontsize=9)

    plt.xlabel("Latent Dimension 1")
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, "latent_space_1d.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] 1D latent space plot saved to: {plot_path}")

    # ======================================================
    # Mesh generation
    # ======================================================
    print("[INFO] Generating meshes for each latent...")
    mesh_paths = []

    for i, latent in enumerate(latents):
        meshes = model.visualize_a_shape(
            key=sorted_keys[i],
            latent=latent,
            grid_res=96,
            clamp_dist=0.1,
            save_suffix=f"interp_{i:02d}",
        )

        if meshes:
            mesh_path = os.path.join(MESH_DIR, f"interp_{i:02d}.ply")
            meshes[0].export(mesh_path)
            mesh_paths.append(mesh_path)
            print(f"[INFO] Saved mesh: {mesh_path}")
        else:
            print(f"[WARN] No mesh output for step {i}")

    print("[INFO] Mesh generation complete.")

    # ======================================================
    # Render animation
    # ======================================================
    print("[INFO] Rendering animation...")

    frames = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path)
        if hasattr(mesh, "geometry") and isinstance(mesh.geometry, dict):
            mesh = trimesh.util.concatenate([
                geom for geom in mesh.geometry.values()
                if hasattr(geom, "vertices") and hasattr(geom, "faces")
            ])

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.add_collection3d(Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolor='lightblue',
            edgecolor='k',
            linewidth=0.1,
            alpha=1.0
        ))

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.view_init(elev=25, azim=30)
        ax.set_box_aspect([1, 1, 1])
        plt.axis('off')

        frame_path = mesh_path.replace(".ply", ".png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(EXPERIMENT_ROOT, "sliding_sphere.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)
    print(f"[INFO] Animation saved: {gif_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

