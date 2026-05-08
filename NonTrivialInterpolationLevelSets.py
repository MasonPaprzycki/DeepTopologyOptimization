import os
import torch
import numpy as np
import matplotlib.pyplot as plt

os.environ["PYOPENGL_PLATFORM"] = "egl"

from skimage import measure
import trimesh
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

import Model
from VisualizeAnalyticSDF import visualize_analytic_sdf


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    REPO_ROOT = os.getcwd()
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", "NonTrivialInterpolation_3Shapes")

    PLOT_DIR = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR = os.path.join(EXPERIMENT_ROOT, "meshes")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)

    # Analytic level-set functions
    R = 0.5

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

    # Scene dictionary
    scenes = {
        "torus": {0: (analytic_torus_sdf, [])},
        "rounded_box": {0: (wavey_rounded_box_sdf, [])},
        "dented_sphere": {0: (dented_sphere_sdf, [])},
    }

    # Visualization
    visualize_analytic_sdf(analytic_torus_sdf, "torus", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(wavey_rounded_box_sdf, "rounded_box", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(dented_sphere_sdf, "dented_sphere", EXPERIMENT_ROOT, grid_res=128)

    # Train model (3D latent space)
    model = Model.Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="NonTrivialInterpolation_3Shapes",
        scenes=scenes,
        latent_dim=3,
        num_epochs=10000,
        samples_per_scene=500_000,
        domain_radius=1.0,
        skip_layer=4,
        training_clamp_dist=None,
        sample_clamp_dist=0.1,
        regularize_latent=True,
        soft_latent=False,
        train_until_convergence=True,
        patience=3,
        min_delta=0.01,
        stochastic_distribution=False
    )

    print("[INFO] Training model...")
    model.train()
    print("[INFO] Training complete.")

    # Grid setup
    GRID_RES = 64

    xs = np.linspace(-1, 1, GRID_RES)
    ys = np.linspace(-1, 1, GRID_RES)
    zs = np.linspace(-1, 1, GRID_RES)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)
    grid_pts = torch.tensor(grid.reshape(-1, 3)).float()

    latent_dict = {}
    canonical_voxels = {}

    # Gaussian weighted error (kept for analysis consistency)
    def gaussian_weighted_error(latent, sdf_true, sigma, N=100000):
        pts = torch.rand(N, 3, device=device) * 2 - 1

        with torch.no_grad():
            f_true = sdf_true(pts).squeeze()
            f_pred = model.compute_sdf_from_latent(latent_vector=latent, xyz=pts)

        w = torch.exp(-(f_true**2) / (2 * sigma**2))

        num = torch.sum(w * (f_true - f_pred) ** 2)
        den = torch.sum(w)

        return (num / den).item()

    sigma = 2.0 / 3.0

    analytic_sdfs = {
        "torus": analytic_torus_sdf,
        "rounded_box": wavey_rounded_box_sdf,
        "dented_sphere": dented_sphere_sdf
    }

    sdf_errors = {}

    # Extract trained latents
    for name, scene in model.trained_scenes.items():

        z = scene.latent_vector.detach()
        short_name = name.split(f"{model.model_name.lower()}_")[-1]

        latent_dict[short_name] = z

        sdf_errors[short_name] = gaussian_weighted_error(
            z,
            analytic_sdfs[short_name],
            sigma=sigma
        )

        canonical_voxels[short_name] = model.sdf_voxel(
            latent=z,
            grid_pts=grid_pts,
            GRID_RES=GRID_RES
        )

        mesh = model.reconstruct_mesh(
            latent=z,
            name=f"canonical_{name}",
            grid_pts=grid_pts,
            GRID_RES=GRID_RES
        )

        if mesh is None:
            continue

        mesh.export(os.path.join(MESH_DIR, f"canonical_{short_name}.ply"))

        img = Model.render_mesh_isometric_pil(mesh)

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), name, fill=(0, 0, 0), font=font)

        img.save(os.path.join(PLOT_DIR, f"canonical_{name}.png"))

    print(f"Mean Gaussian weighted error: {np.mean(list(sdf_errors.values())):.6f}")

    # Topology utilities
    def topo_sim(A, B):
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        return 0 if union == 0 else inter / union

    def leakage(latent, intended):

        vox = model.sdf_voxel(
            latent=latent,
            grid_pts=grid_pts,
            GRID_RES=GRID_RES
        )

        sims = {
            k: topo_sim(vox, v)
            for k, v in canonical_voxels.items()
        }

        # only compare against valid canonical classes
        intended_valid = [k for k in intended if k in sims]
        unintended = [k for k in sims if k not in intended_valid]

        # centroid or degenerate case
        if len(intended_valid) == 0 or len(unintended) == 0:
            return 0.0

        intended_max = max(sims[k] for k in intended_valid)
        unintended_max = max(sims[k] for k in unintended)

        return max(0.0, unintended_max - intended_max)

    # Interpolation setup (3D latent space)
    segments = [
        ("torus", "rounded_box", "Torus → Rounded Box"),
        ("torus", "dented_sphere", "Torus → Dented Sphere"),
        ("rounded_box", "dented_sphere", "Rounded Box → Dented Sphere"),
    ]

    centroid_latent = torch.stack(list(latent_dict.values()), dim=0).mean(dim=0)
    segments.append(("centroid", "centroid", "Latent Superposition (Centroid)"))

    def get_latent(name):
        return centroid_latent if name == "centroid" else latent_dict[name]

    # Sampling
    N_SAMPLES = 21
    ts = torch.linspace(0, 1, N_SAMPLES)

    leakage_log = {}
    total_leakage_log = {}

    views = [(-45, 15), (-30, 90), (30, 210)]

    view_frames = {i: [] for i in range(len(views))}
    combined_frames = []

    # Interpolation loop
    for a, b, label in segments:

        curve = []

        z_a = get_latent(a)
        z_b = get_latent(b)

        for step, t in enumerate(ts):

            name = f"interp_{a}_{b}_{step:02d}"

            if a == "centroid" and b == "centroid":
                z = centroid_latent
            else:
                z = (1 - t) * z_a + t * z_b

            leak = leakage(z, {a, b})
            curve.append(leak)

            mesh = model.reconstruct_mesh(
                latent=z,
                grid_pts=grid_pts,
                GRID_RES=GRID_RES,
                name=name
            )

            if mesh is None:
                continue

            mesh.export(os.path.join(MESH_DIR, f"{name}.ply"))

            step_imgs = []

            for view_id, (pitch, yaw) in enumerate(views):

                img = Model.render_mesh_isometric_pil(
                    mesh,
                    pitch_deg=pitch,
                    yaw_deg=yaw
                )

                draw = ImageDraw.Draw(img)

                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                draw.text((10, 10), label, fill=(0, 0, 0), font=font)
                draw.text((10, 35), f"t={float(t):.3f}", fill=(0, 0, 0), font=font)
                draw.text((10, 60), f"Leak={leak:.4f}", fill=(0, 0, 0), font=font)

                frame_path = os.path.join(PLOT_DIR, f"{name}_view{view_id}.png")
                img.save(frame_path)

                view_frames[view_id].append(frame_path)
                step_imgs.append(img)

            stitched = Image.new(
                "RGB",
                (sum(im.width for im in step_imgs), max(im.height for im in step_imgs))
            )

            x_offset = 0
            for im in step_imgs:
                stitched.paste(im, (x_offset, 0))
                x_offset += im.width

            stitched_path = os.path.join(PLOT_DIR, f"{name}_stitched.png")
            stitched.save(stitched_path)
            combined_frames.append(stitched_path)

        ys = np.asarray(curve, dtype=np.float64)

        if len(ys) < 3:
            total = float(np.sum(ys))
        else:
            if len(ys) % 2 == 0:
                ys = ys[:-1]

            h = 1.0 / (len(ys) - 1)

            total = (h / 3.0) * (
                ys[0] + ys[-1]
                + 4.0 * np.sum(ys[1:-1:2])
                + 2.0 * np.sum(ys[2:-2:2])
            )

        leakage_log[label] = curve
        total_leakage_log[label] = total

        print(f"{label:<40} | total leakage = {total:.6f}")

    # GIF generation
    for view_id, frames in view_frames.items():
        gif_path = os.path.join(PLOT_DIR, f"view_{view_id}_interpolation.gif")
        imgs = [np.asarray(imageio.imread(f)) for f in frames]
        imageio.mimsave(gif_path, imgs, duration=0.12)
        print(f"[INFO] Saved {gif_path}")

    stitched_imgs = [np.asarray(imageio.imread(f)) for f in combined_frames]

    stitched_gif = os.path.join(PLOT_DIR, "combined_views.gif")
    imageio.mimsave(stitched_gif, stitched_imgs, duration=0.12)

    print(f"[INFO] Saved {stitched_gif}")

    # Global stats
    mean_leakage = np.mean(list(total_leakage_log.values()))

    print(f"Mean total leakage : {mean_leakage:.6f}")

    # Plot curves
    plt.figure(figsize=(8,5))

    for label, curve in leakage_log.items():
        plt.plot(ts.numpy(), curve, label=label)

    plt.xlabel("Interpolation parameter")
    plt.ylabel("Shape leakage")
    plt.title("Latent interpolation leakage")

    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(PLOT_DIR, "leakage_curves.png"))
    plt.close()

    print(f"Mean total leakage: {np.mean(list(total_leakage_log.values())):.6f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()