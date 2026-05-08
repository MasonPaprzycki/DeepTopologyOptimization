import os
import numpy as np
import torch
import matplotlib
import imageio.v2 as imageio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

from Model import Model


# Utility logging
def log(msg):
    print(f"[LOG] {msg}")


# Analytic sphere SDF
def analytic_sphere_sdf(xyz, cx, cy, r):

    x = xyz[:,0] - cx
    y = xyz[:,1] - cy
    z = xyz[:,2]

    return torch.sqrt(x**2 + y**2 + z**2) - r

# Scene factory
def make_scene(cx, cy, r):

    def sdf_fn(xyz, params=None):
        return analytic_sphere_sdf(xyz, cx, cy, r).unsqueeze(1)

    return sdf_fn

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device = {device}")

    REPO_ROOT = "/content"
    EXPERIMENT_NAME = "SlidingSphere2D_latent1D"

    EXP_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)
    FRAME_DIR = os.path.join(EXP_ROOT, "frames")
    PLOT_DIR = os.path.join(EXP_ROOT, "plots")

    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    log(f"experiment root = {EXP_ROOT}")

    # Domain
    L = 1.6
    r = 0.5

    log("domain initialized")
    log(f"domain size = {L}")
    log(f"sphere radius = {r}")

    # Scene grid
    n = 2

    x_nodes = np.linspace(-L/2, L/2, n)
    y_nodes = np.linspace(-L/2, L/2, n)

    scenes = {}

    for i, x in enumerate(x_nodes):
        for j, y in enumerate(y_nodes):

            scenes[f"sphere_{i}_{j}"] = {
                0 : (make_scene(x, y, r), [])
            }

    log(f"scene grid created ({n} x {n})")
    log(f"total scenes = {len(scenes)}")

    # Model
    model_name = "SlidingSphereModel2D_latent1D"

    model = Model(
        base_directory = EXP_ROOT,
        model_name = model_name,
        scenes = scenes,
        latent_dim = 1,
        num_epochs = 7000,
        samples_per_scene = 5000,
        training_clamp_dist = None,
        sample_clamp_dist = 2,
        skip_layer = 4,
        regularize_latent = True,
        train_until_convergence = True,
        soft_latent = False,
        patience = 1,
        min_delta = 0.05
    )

    log("model initialized")

    # Training grid
    grid_res = 128

    xv = np.linspace(-1.8, 1.8, grid_res)
    yv = np.linspace(-1.8, 1.8, grid_res)
    zv = np.linspace(-1.8, 1.8, grid_res)

    xx, yy, zz = np.meshgrid(xv, yv, zv)

    xyz_np = np.stack([xx, yy, zz], axis=-1)

    xyz = (
        torch.tensor(xyz_np, dtype=torch.float32)
        .reshape(-1, 3)
        .to(device)
    )

    log("starting training")
    model.train(grid=xyz)
    log("training complete")

    # Extract latent codes
    latent_grid = np.zeros((n, n, 1))

    for i in range(n):
        for j in range(n):

            key = f"{model_name.lower()}_sphere_{i}_{j}"

            latent_grid[i, j] = (
                model.trained_scenes[key]
                .get_latent_vector()
                .cpu()
                .numpy()
            )

    log("latent vectors extracted")

    # Corner latents
    z00 = latent_grid[0,0,0]
    z10 = latent_grid[1,0,0]
    z01 = latent_grid[0,1,0]
    z11 = latent_grid[1,1,0]

    print("\n[LATENTS]")
    print("z00 =", z00)
    print("z10 =", z10)
    print("z01 =", z01)
    print("z11 =", z11)

    # Latent visualization
    plt.figure(figsize=(10,4))

    latent_vals = [
        z00,
        z10,
        z01,
        z11
    ]

    labels = [
        "z00",
        "z10",
        "z01",
        "z11"
    ]

    y_positions = [0,0,0,0]

    plt.scatter(latent_vals, y_positions, s=120)

    for x, y, label in zip(latent_vals, y_positions, labels):

        plt.text(
            x,
            y + 0.02,
            label,
            fontsize=12,
            ha="center"
        )

    plt.yticks([])
    plt.xlabel("latent axis")
    plt.title("1D latent collapse")

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_DIR, "latent_positions.png"),
        dpi=200
    )

    plt.close()

    log("latent plot saved")

    # Evaluation grid
    grid_res = 256

    xv = np.linspace(-1.8, 1.8, grid_res)
    yv = np.linspace(-1.8, 1.8, grid_res)

    xx, yy = np.meshgrid(xv, yv)

    xyz_np = np.stack(
        [xx, yy, np.zeros_like(xx)],
        axis=-1
    )

    xyz = (
        torch.tensor(xyz_np, dtype=torch.float32)
        .reshape(-1, 3)
        .to(device)
    )

    # SDF evaluation helper
    def eval_sdf(latent):

        latent_tensor = torch.tensor(
            [latent],
            dtype=torch.float32,
            device=device
        )

        sdf = model.compute_sdf_from_latent(
            latent_vector=latent_tensor,
            xyz=xyz
        )

        if sdf.dim() == 2:
            sdf = sdf[:,0]

        return sdf.cpu().numpy().reshape(grid_res, grid_res)

    # Visualization colormap
    cdict = {
        "red":[(0,0,0),(0.5,1,1),(1,1,1)],
        "green":[(0,0,0),(0.5,1,1),(1,0,0)],
        "blue":[(0,1,1),(0.5,1,1),(1,0,0)]
    }

    cmap = mcolors.LinearSegmentedColormap("sdf", cdict)

    norm = TwoSlopeNorm(
        vmin=-0.6,
        vcenter=0,
        vmax=0.6
    )

    # 1D latent traversal
    log("generating latent traversal GIF")

    N = 80

    latent_min = min(latent_vals)
    latent_max = max(latent_vals)

    traversal = np.linspace(latent_min, latent_max, N)

    frames = []

    for k, z in enumerate(traversal):

        sdf_img = eval_sdf(z)

        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(
            sdf_img,
            cmap=cmap,
            norm=norm,
            origin="lower"
        )

        fig.colorbar(im, ax=ax)

        ax.set_title(f"latent z = {z:.3f}")

        path = os.path.join(
            FRAME_DIR,
            f"frame_{k:04d}.png"
        )

        plt.savefig(path)
        plt.close()

        frames.append(imageio.imread(path))

    imageio.mimsave(
        os.path.join(EXP_ROOT, "latent_traversal.gif"),
        frames,
        duration=0.05
    )

    log("GIF saved")
    log("experiment complete")


if __name__ == "__main__":
    main()