import os
import numpy as np
import torch
import matplotlib
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from Model import Model
import matplotlib.colors as mcolors

# ======================================================
# Experiment Setup
# ======================================================
def main(): 
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(SCRIPT_DIR)

    EXPERIMENT_NAME = "SlidingSphere2D_latent2D"
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
    # Analytic Sphere SDF
    # ======================================================
    def analytic_sphere_sdf(xyz, params=None):
        cx = 0.0 if params is None else params.get("cx", 0.0)
        cy = 0.0 if params is None else params.get("cy", 0.0)
        cz = 0.0 if params is None else params.get("cz", 0.0)

        x = xyz[:, 0] - cx
        y = xyz[:, 1] - cy
        z = xyz[:, 2] - cz

        radius = 0.5
        return (torch.sqrt(x**2 + y**2 + z**2) - radius).unsqueeze(1)

    # ======================================================
    # Generate 2D sphere scenes
    # ======================================================
    def make_sphere_scene(cx: float, cy: float):
        def sdf_fn(xyz, params=None):
            return analytic_sphere_sdf(xyz, params={"cx": cx, "cy": cy})
        return sdf_fn

    num_scenes_per_axis = 12
    x_positions = np.linspace(-0.8, 0.8, num_scenes_per_axis)
    y_positions = np.linspace(-0.8, 0.8, num_scenes_per_axis)

    scenes = {}
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            key = f"sphere_{i}_{j}"
            scenes[key] = {0: (make_sphere_scene(x, y), [])}

    print(f"[INFO] Created {len(scenes)} 2D latent scenes")

    # ======================================================
    # Train Model with 2D latent
    # ======================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="SlidingSphereModel2D_latent2D",
        scenes=scenes,
        latent_dim=2,  # now 2D latent vector
        num_epochs=3000, # roughly how long it takes to converge on this task
        training_clamp_dist=None,
        sample_clamp_dist=0.5,
        latent_injection_layer=4,
        regularize_latent=False,
        soft_latent=True,
        
    )

    print("[INFO] Training model...")
    model.train()
    print("[INFO] Training complete.")

    print(f"[INFO] Loaded model with {len(model.trained_scenes)} latent vectors")

    # -------------------------------
    # Get latent extremes (corner latents)
    # -------------------------------
    latents = torch.stack([scene.get_latent_vector() for scene in model.trained_scenes.values()]).float()
    latent_min = latents.min(dim=0).values
    latent_max = latents.max(dim=0).values
    print("[INFO] Latent min/max:", latent_min, latent_max)

    # -------------------------------
    # Build 2D evaluation grid
    # -------------------------------
    grid_res = 256
    xv = np.linspace(-1.8, 1.8, grid_res)
    yv = np.linspace(-1.8, 1.8, grid_res)
    xx, yy = np.meshgrid(xv, yv)
    xyz_np = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    xyz = torch.tensor(xyz_np, dtype=torch.float32).reshape(-1, 3).to(device)

    # -------------------------------
    # SDF evaluation
    # -------------------------------
    def eval_sdf(latent_vec):
        sdf = model.compute_sdf_from_latent(latent_vector=latent_vec, xyz=xyz)
        if sdf.dim() == 2:
            sdf = sdf[:, 0]
        return sdf.cpu().numpy().reshape(grid_res, grid_res)

    # -------------------------------
    # Colormap
    # -------------------------------
    cdict = {
        "red":   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "green": [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
        "blue":  [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
    }
    custom_cmap = mcolors.LinearSegmentedColormap("sdf_custom", cdict)
    norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)

    # -------------------------------
    # Latent sweep (bilinear interpolation of corners)
    # -------------------------------
    interp_steps = 25
    frames = []

    print("[INFO] Generating 2D latent-sweep frames...")

    tx_vals = torch.linspace(0.0, 1.0, interp_steps)
    ty_vals = torch.linspace(0.0, 1.0, interp_steps)

    # Define corner latents
    corners = {
        "bottom_left": latent_min,
        "bottom_right": torch.tensor([latent_max[0], latent_min[1]]),
        "top_left": torch.tensor([latent_min[0], latent_max[1]]),
        "top_right": latent_max
    }

    for i, t_x in enumerate(tx_vals):
        for j, t_y in enumerate(ty_vals):
            # Bilinear interpolation in latent space
            latent_vec = (
                (1 - t_x) * (1 - t_y) * corners["bottom_left"] +
                t_x * (1 - t_y) * corners["bottom_right"] +
                (1 - t_x) * t_y * corners["top_left"] +
                t_x * t_y * corners["top_right"]
            )

            sdf_img = eval_sdf(latent_vec)

            plt.figure(figsize=(5, 5))
            plt.imshow(
                sdf_img,
                extent=(-1.8, 1.8, -1.8, 1.8),
                cmap=custom_cmap,
                norm=norm,
                origin="lower"
            )
            plt.colorbar(label="SDF")
            plt.title(f"Latent: [{latent_vec[0]:.2f}, {latent_vec[1]:.2f}]")
            plt.xlabel("x")
            plt.ylabel("y")

            frame_path = os.path.join(FRAME_DIR, f"latent_{i:02d}_{j:02d}.png")
            plt.savefig(frame_path, dpi=120)
            plt.close()
            frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(EXPERIMENT_ROOT, "latent2D_superposition.gif")
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"[INFO] Saved GIF: {gif_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
