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
from matplotlib.colors import Normalize

# ======================================================
# Experiment Setup
# ======================================================
def main(): 
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(SCRIPT_DIR)

    EXPERIMENT_NAME = "SlidingSphere1D_latent1D"
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

    num_scenes = 10
    positions = np.linspace(-0.8, 0.8, num_scenes)
   

    scenes = {}
    for i, x in enumerate(positions):
        key = f"sphere_{i}"
        scenes[key] = {0: (make_sphere_scene(x, 0.0), [])}



    print(f"[INFO] Created {len(scenes)} 1D latent scenes")

    # ======================================================
    # Train Model with 1D latent
    # ======================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "SlidingSphereModel1D_latent1D"

    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="SlidingSphereModel1D_latent1D",
        scenes=scenes,
        scenes_per_batch=1,
        latent_dim=1,  # now 1D latent vector
        num_epochs=1000, # roughly how long it takes to converge on this task
        training_clamp_dist=0.1,
        sample_clamp_dist=0.1,
        
        skip_layer=4,
        regularize_latent=False,
        soft_latent=False,
        weight_norm=False
    )

    print("[INFO] Training model...")
    model.train()
    print("[INFO] Training complete.")
 

    print(f"[INFO] Loaded model with {len(model.trained_scenes)} latent vectors")

    # -------------------------------
    # Get latent extremes (corner latents)
    # -------------------------------
    latent_list = []
    print("trained scenes:", model.trained_scenes.keys())
    for i in range(num_scenes):
        key = f"{str.lower(model_name)}_sphere_{i}"
        latent_list.append(model.trained_scenes[key].get_latent_vector())

    latents = torch.stack(latent_list).float()
  
    latent_min = latents.min(dim=0).values
    latent_max = latents.max(dim=0).values
    print("[INFO] Latent min/max:", latent_min, latent_max)

    # ======================================================
    # Latent ordering analysis
    # ======================================================

    positions_np = np.array(positions)

    latent_values = latents.cpu().numpy().flatten()

    # --------------------------------------
    # Determine dominant direction
    # --------------------------------------
    corr = np.corrcoef(positions_np, latent_values)[0,1]
    direction = 1 if corr > 0 else -1

    print("[INFO] Position/latent correlation:", corr)
    print("[INFO] Dominant ordering direction:", "positive" if direction==1 else "negative")

    # --------------------------------------
    # Sort by spatial position
    # --------------------------------------
    order = np.argsort(positions_np)

    x_sorted = positions_np[order]
    z_sorted = latent_values[order]

    N = len(z_sorted)

    disorder_percent = []

    for i in range(N):

        zi = z_sorted[i]
        violations = 0
        total = 0

        for j in range(i+1, N):

            zj = z_sorted[j]

            if direction * (zj - zi) < 0:
                violations += 1

            total += 1

        if total > 0:
            disorder_percent.append(100 * violations / total)
        else:
            disorder_percent.append(0)

    disorder_percent = np.array(disorder_percent)

    print("[INFO] Mean disorder:", disorder_percent.mean(), "%")



    # -------------------------------
    # Latent disorder curve
    # -------------------------------

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(x_sorted, disorder_percent, marker='o')

    ax.set_xlabel("Sphere Position")
    ax.set_ylabel("% Latents Out of Place")
    ax.set_title("Latent Ordering Disorder Across Dataset")

    ax.set_ylim(0,100)
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        os.path.join(PLOT_DIR, "latent_disorder_curve.png"),
        dpi=150
    )

    plt.close(fig)

    # -------------------------------
    # Build 1D evaluation grid
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

    print("[INFO] Generating 1D latent-sweep frames...")

    t_vals = torch.linspace(0.0, 1.0, interp_steps)
  

    # Define edge latents
    edges = {
    "left": latent_min.to(device),
    "right": latent_max.to(device),
    }


    for i, t in enumerate(t_vals):
        latent_vec = (1 - t) * edges["left"] + t * edges["right"]

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
        plt.title(f"Latent: {latent_vec[0]:.2f}")

        plt.xlabel("x")
        plt.ylabel("y")

        frame_path = os.path.join(FRAME_DIR, f"latent_{i:02d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close()
        frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(EXPERIMENT_ROOT, "latent1D_superposition.gif")
    imageio.mimsave(gif_path, frames, duration=0.05)
    print(f"[INFO] Saved GIF: {gif_path}")

    latent_values = latents.cpu().numpy().flatten()

    # ======================================================
    # Latent Structure Metrics
    # ======================================================

    from scipy.stats import spearmanr, kendalltau
    from sklearn.linear_model import LinearRegression

    print("\n[INFO] ---- Latent Structure Metrics ----")

    # -------------------------------
    # Spearman Rank Correlation
    # -------------------------------

    spearman_rho, spearman_p = spearmanr(positions_np, latent_values)

    print(f"[METRIC] Spearman rho: {spearman_rho:.4f}")
    print(f"[METRIC] Spearman p-value: {spearman_p:.4e}")

    # -------------------------------
    # Kendall Tau
    # -------------------------------

    kendall_tau, kendall_p = kendalltau(positions_np, latent_values)

    print(f"[METRIC] Kendall tau: {kendall_tau:.4f}")
    print(f"[METRIC] Kendall p-value: {kendall_p:.4e}")

    # -------------------------------
    # Linear Regression R^2
    # -------------------------------

    reg = LinearRegression()
    reg.fit(latent_values.reshape(-1,1), positions_np)

    r2 = reg.score(latent_values.reshape(-1,1), positions_np)

    print(f"[METRIC] Linear regression R^2 (position from latent): {r2:.4f}")

    # -------------------------------
    # Latent Dot Plot 
    # -------------------------------

    plt.figure(figsize=(6,4))

    plt.scatter(positions_np, latent_values, s=90)

    plt.xlabel("True Sphere Position")
    plt.ylabel("Latent Value")
    plt.title("Latent Vectors corresponding to Sphere Positions")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "latent_dotplot.png"), dpi=150)
    plt.close()

    # -------------------------------
    # Geometry Lipschitz Analysis
    # -------------------------------

    print("\n[INFO] Computing latent Lipschitz geometry ratios...")

    sdf_cache = []

    for z in latent_values:
        latent_vec = torch.tensor([z], dtype=torch.float32).to(device)
        sdf_cache.append(eval_sdf(latent_vec).flatten())

    ratios = []

    for i in range(len(latent_values)):
        for j in range(i+1, len(latent_values)):

            geom_diff = np.linalg.norm(sdf_cache[i] - sdf_cache[j])
            latent_diff = abs(latent_values[i] - latent_values[j])

            if latent_diff > 1e-8:
                ratios.append(geom_diff / latent_diff)

    ratios = np.array(ratios)

    print(f"[METRIC] Geometry/Latent ratio mean: {ratios.mean():.4f}")
    print(f"[METRIC] Geometry/Latent ratio std:  {ratios.std():.4f}")
    print(f"[METRIC] Geometry/Latent ratio min:  {ratios.min():.4f}")
    print(f"[METRIC] Geometry/Latent ratio max:  {ratios.max():.4f}")

    # -------------------------------
    # Lipschitz Histogram
    # -------------------------------

    plt.figure(figsize=(6,4))

    plt.hist(ratios, bins=20)

    plt.xlabel("||geometry_i - geometry_j|| / ||latent_i - latent_j||")
    plt.ylabel("Frequency")

    plt.title("Geometry Sensitivity to Latent Changes")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "latent_lipschitz_histogram.png"), dpi=150)
    plt.close()

    print("[INFO] Latent structure analysis complete.")



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
