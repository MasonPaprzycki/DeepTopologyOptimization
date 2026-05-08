import numpy as np
import torch
import matplotlib
import trimesh
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from Model import Model
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression


# ============================================================
# Experiment
# ============================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "SlidingSphereModel1D_latent1D"

    REPO_ROOT = "/content"
    EXPERIMENT_NAME = "SlidingSphere1D_latent1D"
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

    FRAME_DIR = os.path.join(EXPERIMENT_ROOT, "frames_latents")
    PLOT_DIR = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR = os.path.join(EXPERIMENT_ROOT, "Meshes")

    for d in [EXPERIMENT_ROOT, FRAME_DIR, PLOT_DIR, MESH_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")


    # ============================================================
    # Analytic SDF
    # ============================================================

    def analytic_sphere_sdf(xyz, cx=0.0):
        x = xyz[:, 0] - cx
        y = xyz[:, 1]
        z = xyz[:, 2]
        r = 0.5
        return (torch.sqrt(x**2 + y**2 + z**2) - r).unsqueeze(1)


    def make_scene(cx):
        return lambda xyz, params=None: analytic_sphere_sdf(xyz, cx)


    # ============================================================
    # Dataset
    # ============================================================

    num_scenes = 3
    positions = np.linspace(-0.8, 0.8, num_scenes)

    scenes = {}
    for i, x in enumerate(positions):
        scenes[f"sphere_{i}"] = {0: (make_scene(x), [])}

    print(f"[INFO] Scenes: {len(scenes)}")


    # ============================================================
    # Model
    # ============================================================

    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name=model_name,
        scenes=scenes,
        scenes_per_batch=1,
        latent_dim=1,
        num_epochs=4000,
        training_clamp_dist=0.1,
        samples_per_scene=5000,
        sample_clamp_dist=0.1,
        skip_layer=4,
        regularize_latent=True,
        soft_latent=False,
        weight_norm=False,
        train_until_convergence=True,
        stochastic_distribution=True, 
        patience=1,
        window=50,
        min_delta=0.05
    )

    grid_res = 128

    x = np.linspace(-1.8,1.8,grid_res)
    y = np.linspace(-1.8,1.8,grid_res)
    z= np.linspace(-1.8,1.8,grid_res)


    X,Y,Z = np.meshgrid(x,y,z)

    xyz_np = np.stack([X,Y,Z],axis=-1)

    xyz = torch.tensor(xyz_np,dtype=torch.float32).reshape(-1,3).to(device)

    print("[INFO] Training model...")
    model.train(grid=xyz)
    print("[INFO] Training complete.")



    # ============================================================
    # Latents
    # ============================================================

    latent_list = []
    for i in range(num_scenes):
        key = f"{model_name.lower()}_sphere_{i}"
        latent_list.append(model.trained_scenes[key].get_latent_vector())

    latents = torch.stack(latent_list).float()
    latent_min = latents.min(0).values.to(device)
    latent_max = latents.max(0).values.to(device)

    positions_np = np.array(positions)
    latent_values = latents.cpu().numpy().flatten()

    # ============================================================
    # Latent vs Sphere Position Plot
    # ============================================================

    plt.figure(figsize=(6,4))

    plt.scatter(positions_np, latent_values, s=80)

    # optional: line fit for visualization
    reg = LinearRegression()
    reg.fit(positions_np.reshape(-1,1), latent_values)
    x_line = np.linspace(positions_np.min(), positions_np.max(), 100)
    y_line = reg.predict(x_line.reshape(-1,1))

    plt.plot(x_line, y_line)

    plt.xlabel("Sphere Position (x)")
    plt.ylabel("Latent Value (z)")
    plt.title("Learned Latent Coordinate vs Sphere Position")

    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "latent_vs_position.png"))
    plt.close()


    # ============================================================
    # Latent structure analysis
    # ============================================================

    corr = np.corrcoef(positions_np, latent_values)[0, 1]
    direction = 1 if corr > 0 else -1

    order = np.argsort(positions_np)
    z_sorted = latent_values[order]

    disorder = []
    for i in range(len(z_sorted)):
        v = 0
        total = 0
        for j in range(i+1, len(z_sorted)):
            if direction * (z_sorted[j] - z_sorted[i]) < 0:
                v += 1
            total += 1
        disorder.append(100 * v / total if total > 0 else 0)

    disorder = np.array(disorder)

    plt.figure(figsize=(6,4))
    plt.plot(positions_np[order], disorder, marker='o')
    plt.title("Latent Ordering Disorder")
    plt.xlabel("Position")
    plt.ylabel("% Disorder")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "latent_disorder_curve.png"))
    plt.close()


    # ============================================================
    # Grid
    # ============================================================

    grid_res = 256
    xv = np.linspace(-1.8, 1.8, grid_res)
    yv = np.linspace(-1.8, 1.8, grid_res)
    xx, yy = np.meshgrid(xv, yv)
    xyz_np = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)

    xyz = torch.tensor(xyz_np, dtype=torch.float32).reshape(-1, 3).to(device)


    def eval_sdf(latent_vec):
        out = model.compute_sdf_from_latent(
            latent_vector=latent_vec,
            xyz=xyz
        )
        return out.detach().cpu().numpy().reshape(grid_res, grid_res)


    # ============================================================
    # Colormap
    # ============================================================

    cdict = {
        "red": [(0,0,0),(0.5,1,1),(1,1,1)],
        "green": [(0,0,0),(0.5,1,1),(1,0,0)],
        "blue": [(0,1,1),(0.5,1,1),(1,0,0)]
    }

    cmap = mcolors.LinearSegmentedColormap("sdf", cdict)
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)


    # ============================================================
    # Latent interpolation parameterization
    # ============================================================

    interp_steps = 50
    t_vals = np.linspace(0.0, 1.0, interp_steps)

    z0 = latents[0].to(device)
    z1 = latents[-1].to(device)

    all_errors = []
    E_t = []

    frames_sdf = []
    frames_error = []


    # ============================================================
    # Monte Carlo estimator (definition-matching)
    # ============================================================

    def gaussian_weighted_error(f_pred, f_true, sigma, N=60000):

        x = np.random.uniform(-1.5, 1.5, N)
        y = np.random.uniform(-1.5, 1.5, N)
        z = np.random.uniform(-1.5, 1.5, N)

        pts = np.stack([x, y, z], axis=-1)

        ft = f_true(pts)
        fp = f_pred(pts)

        w = np.exp(-(ft**2) / (2 * sigma**2))

        return np.sum(w * (ft - fp)**2) / np.sum(w)


    sigma = (3.0) / 3.0  # domain-scale rule


    # ============================================================
    # E(t) evaluation
    # ============================================================

    for i, t in enumerate(t_vals):

        # correct O→1 parameterization
        zt = (1 - t) * z0 + t * z1
        cx = (1 - t) * positions.min() + t * positions.max()

        def f_pred(x):
            xt = torch.from_numpy(x).float().to(device)
            return model.compute_sdf_from_latent(
                latent_vector=zt,
                xyz=xt
            ).detach().cpu().numpy().squeeze()

        def f_true(x):
            return np.sqrt((x[:,0]-cx)**2 + x[:,1]**2 + x[:,2]**2) - 0.5

        E = gaussian_weighted_error(f_pred, f_true, sigma)
        E_t.append(E)

        sdf_img = eval_sdf(zt)

        true_img = analytic_sphere_sdf(xyz, cx).cpu().numpy().reshape(grid_res, grid_res)

        all_errors.append(sdf_img - true_img)


    E_t = np.array(E_t)

    # ============================================================
    # Integral over t
    # ============================================================
    import scipy.integrate as integrate
    integral_error = integrate.simpson(E_t, t_vals)
    
    # integral_error = np.trapz(E_t, t_vals)

    print(f"[METRIC] ∫ E(t) dt = {integral_error:.6f}")
    print(f"[METRIC] mean E(t) = {E_t.mean():.6f}")


    # ============================================================
    # Plot curve
    # ============================================================

    plt.figure(figsize=(6,4))
    plt.plot(t_vals, E_t, marker='o')
    plt.title("Reconstruction Error E(t)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "reconstruction_error_curve.png"))
    plt.close()

    # ============================================================
    # GIF generation (with colorbars)
    # ============================================================

    all_errors = np.stack(all_errors)

    for i, t in enumerate(t_vals):

        zt = (1 - t) * z0 + t * z1
        sdf_img = eval_sdf(zt)

        # ---------------------------
        # SDF frame
        # ---------------------------
        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(
            sdf_img,
            cmap=cmap,
            norm=norm,
            origin="lower"
        )

        ax.set_title(f"SDF  t={t:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("SDF value")

        path = os.path.join(FRAME_DIR, f"sdf_{i:02d}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        frames_sdf.append(imageio.imread(path))


        # ---------------------------
        # Error frame
        # ---------------------------
        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(
            all_errors[i],
            cmap=cmap,
            norm=norm,
            origin="lower"
        )

        ax.set_title(f"Error  t={t:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("SDF Error")

        path = os.path.join(FRAME_DIR, f"err_{i:02d}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        frames_error.append(imageio.imread(path))


    imageio.mimsave(
        os.path.join(EXPERIMENT_ROOT, "sdf.gif"),
        frames_sdf,
        duration=0.05
    )

    imageio.mimsave(
        os.path.join(EXPERIMENT_ROOT, "error.gif"),
        frames_error,
        duration=0.05
    )
    # ============================================================
    # Final metrics
    # ============================================================

    print("\n[INFO] ---- Metrics ----")
    print("Spearman:", spearmanr(positions_np, latent_values)[0])
    print("Kendall:", kendalltau(positions_np, latent_values)[0])

    reg = LinearRegression()
    reg.fit(latent_values.reshape(-1,1), positions_np)
    print("R^2:", reg.score(latent_values.reshape(-1,1), positions_np))


if __name__ == "__main__":
    main()