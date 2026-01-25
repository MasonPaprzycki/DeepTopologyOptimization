# ======================================================
# Colab + Drive setup (GIF ONLY)
# ======================================================
from google.colab import drive
drive.mount("/content/drive")

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
    # ---- LOCAL WORKING DIRECTORY (FAST) ----
    LOCAL_ROOT = "/content/SlidingSphere"
    EXPERIMENT_NAME = "SlidingSphere2D_latent2D"
    EXPERIMENT_ROOT = os.path.join(LOCAL_ROOT, EXPERIMENT_NAME)

    FRAME_DIR = os.path.join(EXPERIMENT_ROOT, "frames_latents")
    PLOT_DIR  = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR  = os.path.join(EXPERIMENT_ROOT, "Meshes")

    for d in [
        EXPERIMENT_ROOT,
        FRAME_DIR,
        PLOT_DIR,
        MESH_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Local experiment directory: {EXPERIMENT_ROOT}")

    # ---- DRIVE OUTPUT (SLOW, FINAL ONLY) ----
    DRIVE_GIF_DIR = "/content/drive/MyDrive/SlidingSphereOutputs"
    os.makedirs(DRIVE_GIF_DIR, exist_ok=True)

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
    # Scene generation
    # ======================================================
    def make_sphere_scene(cx, cy):
        def sdf_fn(xyz, params=None):
            return analytic_sphere_sdf(xyz, {"cx": cx, "cy": cy})
        return sdf_fn

    num_scenes_per_axis = 10
    x_positions = np.linspace(-0.8, 0.8, num_scenes_per_axis)
    y_positions = np.linspace(-0.8, 0.8, num_scenes_per_axis)

    scenes = {}
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            scenes[f"sphere_{i}_{j}"] = {0: (make_sphere_scene(x, y), [])}

    print(f"[INFO] Created {len(scenes)} scenes")

    # ======================================================
    # Train model
    # ======================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="SlidingSphereModel2D_latent2D",
        scenes=scenes,
        latent_dim=2,
        num_epochs=4000,
        training_clamp_dist=None,
        sample_clamp_dist=0.5,
        latent_injection_layer=4,
        regularize_latent=False,
        soft_latent=True,
        device=device,
        samples_per_scene=5000
    )

    print("[INFO] Training model...")
    model.train()
    print("[INFO] Training complete.")

    # ======================================================
    # Latent bounds
    # ======================================================
    latents = torch.stack(
        [s.get_latent_vector() for s in model.trained_scenes.values()]
    ).float()

    latent_min = latents.min(dim=0).values
    latent_max = latents.max(dim=0).values

    # ======================================================
    # Evaluation grid
    # ======================================================
    grid_res = 256
    xv = np.linspace(-1.8, 1.8, grid_res)
    yv = np.linspace(-1.8, 1.8, grid_res)
    xx, yy = np.meshgrid(xv, yv)
    xyz = torch.tensor(
        np.stack([xx, yy, np.zeros_like(xx)], axis=-1),
        dtype=torch.float32
    ).reshape(-1, 3).to(device)

    def eval_sdf(latent_vec):
        sdf = model.compute_sdf_from_latent(latent_vec, xyz)
        return sdf[:, 0].cpu().numpy().reshape(grid_res, grid_res)

    # ======================================================
    # Visualization setup
    # ======================================================
    cdict = {
        "red":   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "green": [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
        "blue":  [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
    }
    cmap = mcolors.LinearSegmentedColormap("sdf_custom", cdict)
    norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)

    # ======================================================
    # Latent sweep
    # ======================================================
    interp_steps = 25
    frames = []

    tx_vals = torch.linspace(0.0, 1.0, interp_steps)
    ty_vals = torch.linspace(0.0, 1.0, interp_steps)

    corners = {
        "bl": latent_min,
        "br": torch.tensor([latent_max[0], latent_min[1]]),
        "tl": torch.tensor([latent_min[0], latent_max[1]]),
        "tr": latent_max,
    }

    print("[INFO] Generating frames locally...")

    for i, tx in enumerate(tx_vals):
        for j, ty in enumerate(ty_vals):
            latent = (
                (1 - tx) * (1 - ty) * corners["bl"]
                + tx * (1 - ty) * corners["br"]
                + (1 - tx) * ty * corners["tl"]
                + tx * ty * corners["tr"]
            )

            sdf_img = eval_sdf(latent)

            plt.figure(figsize=(5, 5))
            plt.imshow(
                sdf_img,
                extent=(-1.8, 1.8, -1.8, 1.8),
                cmap=cmap,
                norm=norm,
                origin="lower",
            )
            plt.colorbar(label="SDF")
            plt.title(f"[{latent[0]:.2f}, {latent[1]:.2f}]")

            frame_path = os.path.join(FRAME_DIR, f"latent_{i:02d}_{j:02d}.png")
            plt.savefig(frame_path, dpi=120)
            plt.close()

            frames.append(imageio.imread(frame_path))

    # ======================================================
    # SAVE GIF TO DRIVE (ONLY)
    # ======================================================
    gif_path = os.path.join(
        DRIVE_GIF_DIR,
        "latent2D_superposition.gif"
    )
    imageio.mimsave(gif_path, frames, duration=0.05)

    print(f"[INFO] GIF saved to Google Drive: {gif_path}")


if __name__ == "__main__":
    main()
