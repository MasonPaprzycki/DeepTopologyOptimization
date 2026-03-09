#look into spearmans rank correlation coefficient for measuring monotonic relationship between position and latent value, rather than just linear correlation. This would be more robust to non-linear but still monotonic relationships.

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

    num_scenes_per_axis = 5
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

    model_name= "SlidingSphereModel2D_latent2D"

    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name=model_name,
        scenes=scenes,
        latent_dim=2,  # now 2D latent vector
        num_epochs=1, # roughly how long it takes to converge on this task
        training_clamp_dist=0.1,
        sample_clamp_dist=0.1,
        skip_layer=4,
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
    # -------------------------------
    # Collect latents in grid order
    # -------------------------------

    latent_grid = np.zeros((num_scenes_per_axis, num_scenes_per_axis, 2))

    for i in range(num_scenes_per_axis):
        for j in range(num_scenes_per_axis):
            key = f"{str.lower(model_name)}_sphere_{i}_{j}"
            latent = model.trained_scenes[key].get_latent_vector().cpu().numpy()
            latent_grid[i, j] = latent

    latents = latent_grid.reshape(-1, 2)

    x_positions_np = np.tile(x_positions, num_scenes_per_axis)
    y_positions_np = np.repeat(y_positions, num_scenes_per_axis)


    latent_x = latents[:,0]
    latent_y = latents[:,1]


    corr_x = np.corrcoef(x_positions_np, latent_x)[0,1]
    corr_y = np.corrcoef(y_positions_np, latent_y)[0,1]

    dir_x = 1 if corr_x > 0 else -1
    dir_y = 1 if corr_y > 0 else -1

    print("[INFO] X correlation:", corr_x)
    print("[INFO] Y correlation:", corr_y)

    #x-axis disorder 
    order = np.argsort(x_positions_np)

    x_sorted = x_positions_np[order]
    z_sorted = latent_x[order]

    N = len(z_sorted)

    disorder_x = []

    for i in range(N):

        zi = z_sorted[i]
        violations = 0
        total = 0

        for j in range(i+1, N):

            zj = z_sorted[j]

            if dir_x * (zj - zi) < 0:
                violations += 1

            total += 1

        disorder_x.append(100 * violations / total if total > 0 else 0)

    disorder_x = np.array(disorder_x)

    #y-axis disorder
    order = np.argsort(y_positions_np)

    y_sorted = y_positions_np[order]
    z_sorted = latent_y[order]

    disorder_y = []

    for i in range(N):

        zi = z_sorted[i]
        violations = 0
        total = 0

        for j in range(i+1, N):

            zj = z_sorted[j]

            if dir_y * (zj - zi) < 0:
                violations += 1

            total += 1

        disorder_y.append(100 * violations / total if total > 0 else 0)

    disorder_y = np.array(disorder_y)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # =========================================
    # Combined latent vs true position plot
    # X -> R, Y -> G, constant B
    # =========================================

    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # ------------------------------------------------
    # Flatten all coordinate arrays
    # ------------------------------------------------
    latent_x_flat = latent_x.flatten()
    latent_y_flat = latent_y.flatten()
    x_flat = x_positions_np.flatten()
    y_flat = y_positions_np.flatten()
    N = latent_x_flat.shape[0]

    # ------------------------------------------------
    # Normalize X and Y to [0,1]
    # ------------------------------------------------
    x_norm = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min())
    y_norm = (y_flat - y_flat.min()) / (y_flat.max() - y_flat.min())

    # ------------------------------------------------
    # Build RGB colors
    # R = X, G = Y, B = constant 0.5
    # ------------------------------------------------
    colors = np.stack([x_norm, y_norm, np.full(N, 0.5)], axis=1).astype(np.float32)

    # ------------------------------------------------
    # Latent space
    # ------------------------------------------------
    axes[0].scatter(
        latent_x_flat,
        latent_y_flat,
        c=colors,
        s=40,
        alpha=0.9
    )
    axes[0].set_xlabel("Latent dim 0")
    axes[0].set_ylabel("Latent dim 1")
    axes[0].set_title("Latent Space")
    axes[0].grid(True)
    axes[0].set_aspect("equal")

    # ------------------------------------------------
    # True sphere positions
    # ------------------------------------------------
    sc = axes[1].scatter(
        x_flat,
        y_flat,
        c=colors,
        s=40,
        alpha=0.9
    )
    axes[1].set_xlabel("True X")
    axes[1].set_ylabel("True Y")
    axes[1].set_title("True Sphere Positions")
    axes[1].grid(True)
    axes[1].set_aspect("equal")

    # ------------------------------------------------
    # Create color legend
    # ------------------------------------------------
    divider = make_axes_locatable(axes[1])
    cax_x = divider.append_axes("bottom", size="6%", pad=0.5)
    cax_y = divider.append_axes("left", size="6%", pad=0.5)

    # Horizontal (X) gradient
    grad_x = np.tile(np.linspace(0,1,256), (20,1))
    cax_x.imshow(np.dstack([grad_x, np.zeros_like(grad_x), 0.5*np.ones_like(grad_x)]), origin='lower', aspect='auto')
    cax_x.set_xlabel("X position")
    cax_x.set_yticks([])

    # Vertical (Y) gradient
    grad_y = np.tile(np.linspace(0,1,256)[:,None], (1,20))
    cax_y.imshow(np.dstack([np.zeros_like(grad_y), grad_y, 0.5*np.ones_like(grad_y)]), origin='lower', aspect='auto')
    cax_y.set_ylabel("Y position")
    cax_y.set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"latent_vs_true_combined.png"), dpi=200)
    plt.close()

    #disorder curve x
    plt.figure(figsize=(6,4))
    plt.plot(x_sorted, disorder_x, marker='o')

    plt.xlabel("True X position")
    plt.ylabel("% Latents Out of Place")
    plt.title("X-Axis Latent Disorder")

    plt.ylim(0,100)
    plt.grid(True)

    plt.savefig(os.path.join(PLOT_DIR,"latent_disorder_x.png"), dpi=150)
    plt.close()

    #disorder curve y
    plt.figure(figsize=(6,4))
    plt.plot(y_sorted, disorder_y, marker='o')

    plt.xlabel("True Y position")
    plt.ylabel("% Latents Out of Place")
    plt.title("Y-Axis Latent Disorder")

    plt.ylim(0,100)
    plt.grid(True)

    plt.savefig(os.path.join(PLOT_DIR,"latent_disorder_y.png"), dpi=150)
    plt.close()

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
    device  = latent_min.device
    
    corners = {
    "bottom_left": model.trained_scenes[f"{str.lower(model_name)}_sphere_0_0"].get_latent_vector(),
    "bottom_right": model.trained_scenes[f"{str.lower(model_name)}_sphere_{str(num_scenes_per_axis-1)}_0"].get_latent_vector(),
    "top_left": model.trained_scenes[f"{str.lower(model_name)}_sphere_0_{str(num_scenes_per_axis-1)}"].get_latent_vector(),
    "top_right": model.trained_scenes[f"{str.lower(model_name)}_sphere_{str(num_scenes_per_axis-1)}_{str(num_scenes_per_axis-1)}"].get_latent_vector(),
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


    # ======================================================
    # Latent Structure Metrics (2D case)
    # ======================================================

    from scipy.stats import spearmanr, kendalltau
    from sklearn.linear_model import LinearRegression

    print("\n[INFO] ---- Latent Structure Metrics (2D) ----")

    # -------------------------------
    # Spearman Rank Correlation
    # -------------------------------

    rho_x0, p_x0 = spearmanr(x_positions_np, latent_x)
    rho_x1, p_x1 = spearmanr(x_positions_np, latent_y)

    rho_y0, p_y0 = spearmanr(y_positions_np, latent_x)
    rho_y1, p_y1 = spearmanr(y_positions_np, latent_y)

    print("\n[METRIC] Spearman correlations")

    print(f"X vs latent_dim0: rho={rho_x0:.4f}  p={p_x0:.2e}")
    print(f"X vs latent_dim1: rho={rho_x1:.4f}  p={p_x1:.2e}")

    print(f"Y vs latent_dim0: rho={rho_y0:.4f}  p={p_y0:.2e}")
    print(f"Y vs latent_dim1: rho={rho_y1:.4f}  p={p_y1:.2e}")

    # -------------------------------
    # Kendall Tau (ordering metric)
    # -------------------------------

    tau_x0, p = kendalltau(x_positions_np, latent_x)
    tau_x1, p = kendalltau(x_positions_np, latent_y)

    tau_y0, p = kendalltau(y_positions_np, latent_x)
    tau_y1, p = kendalltau(y_positions_np, latent_y)

    print("\n[METRIC] Kendall tau ordering")

    print(f"X vs latent_dim0: tau={tau_x0:.4f}")
    print(f"X vs latent_dim1: tau={tau_x1:.4f}")

    print(f"Y vs latent_dim0: tau={tau_y0:.4f}")
    print(f"Y vs latent_dim1: tau={tau_y1:.4f}")

    # -------------------------------
    # Linear regression R^2
    # (predict position from latents)
    # -------------------------------

    reg_x = LinearRegression()
    reg_x.fit(latents, x_positions_np)
  
    r2_x = reg_x.score(np.array(latents), x_positions_np)

    reg_y = LinearRegression()
    reg_y.fit(latents, y_positions_np)
    r2_y = reg_y.score(np.array(latents), y_positions_np)

    print("\n[METRIC] Linear regression")

    print(f"R^2 predicting X from latents: {r2_x:.4f}")
    print(f"R^2 predicting Y from latents: {r2_y:.4f}")

    # ======================================================
    # Geometry Lipschitz Analysis (latent vs geometry)
    # ======================================================

    print("\n[INFO] Computing latent Lipschitz geometry ratios...")

    latent_values = latents
    sdf_cache = []

    for z in latent_values:
        latent_vec = torch.tensor(z, dtype=torch.float32).to(device)
        sdf_cache.append(eval_sdf(latent_vec).flatten())

    ratios = []

    for i in range(len(latent_values)):
        for j in range(i+1, len(latent_values)):

            geom_diff = np.linalg.norm(sdf_cache[i] - sdf_cache[j])
            latent_diff = np.linalg.norm(latent_values[i] - latent_values[j])

            if latent_diff > 1e-8:
                ratios.append(geom_diff / latent_diff)

    ratios = np.array(ratios)

    print("\n[METRIC] Geometry / Latent sensitivity")

    print(f"Mean: {ratios.mean():.4f}")
    print(f"Std:  {ratios.std():.4f}")
    print(f"Min:  {ratios.min():.4f}")
    print(f"Max:  {ratios.max():.4f}")

    # -------------------------------
    # Lipschitz Histogram
    # -------------------------------

    plt.figure(figsize=(6,4))

    plt.hist(ratios, bins=25)

    plt.xlabel("||geometry_i - geometry_j|| / ||latent_i - latent_j||")
    plt.ylabel("Frequency")

    plt.title("Geometry Sensitivity to Latent Changes")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "latent_lipschitz_histogram.png"), dpi=150)
    plt.close()

    print("\n[INFO] Latent structure analysis complete.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
