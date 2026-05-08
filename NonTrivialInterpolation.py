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
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", "NonTrivialInterpolation")

    PLOT_DIR = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR = os.path.join(EXPERIMENT_ROOT, "meshes")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)

    # Analytic SDF definitions

    # Utility: smooth union
    def smooth_min(a, b, k):
        h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return torch.lerp(b, a, h) - k * h * (1.0 - h)
    
    # Torus primitives

    def torus_z(p, params=None):
        R = 0.5 if params is None else params.get("R", 0.5)
        r = 0.2 if params is None else params.get("r", 0.2)

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        qx = torch.sqrt(x * x + y * y + 1e-12) - R
        qy = z
        return torch.sqrt(qx * qx + qy * qy + 1e-12) - r


    def torus_x(p, params=None):
        R = 0.5 if params is None else params.get("R", 0.5)
        r = 0.2 if params is None else params.get("r", 0.2)

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        qx = torch.sqrt(y * y + z * z + 1e-12) - R
        qy = x
        return torch.sqrt(qx * qx + qy * qy + 1e-12) - r


    def torus_y(p, params=None):
        R = 0.5 if params is None else params.get("R", 0.5)
        r = 0.2 if params is None else params.get("r", 0.2)

        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        qx = torch.sqrt(x * x + z * z + 1e-12) - R
        qy = y
        return torch.sqrt(qx * qx + qy * qy + 1e-12) - r

    # SDFs
    def ellipsoid_sdf(xyz, params=None):
        a = 0.7 if params is None else params.get("a", 0.7)
        c = 0.4 if params is None else params.get("c", 0.4)

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        px = x / a
        py = y / a
        pz = z / c

        sdf = torch.sqrt(px * px + py * py + pz * pz + 1e-12) - 1.0
        return sdf.unsqueeze(1)


    def torus_sdf(xyz, params=None):
        return torus_z(xyz, params).unsqueeze(1)


    def double_torus_sdf(xyz, params=None):
        R = 0.5 if params is None else params.get("R", 0.5)
        r = 0.18 if params is None else params.get("r", 0.18)
        k = 0.12 if params is None else params.get("k", 0.12)

        d1 = torus_x(xyz, params)
        d2 = torus_z(xyz, params)

        d = smooth_min(d1, d2, k)
        return d.unsqueeze(1)


    def triple_torus_sdf(xyz, params=None):
        R = 0.45 if params is None else params.get("R", 0.45)
        r = 0.16 if params is None else params.get("r", 0.16)
        k = 0.12 if params is None else params.get("k", 0.12)

        d1 = torus_x(xyz, params)
        d2 = torus_y(xyz, params)
        d3 = torus_z(xyz, params)

        d = smooth_min(d1, d2, k)
        d = smooth_min(d, d3, k)

        return d.unsqueeze(1)



    # Scene dictionary
    scenes = {
    "ellipsoid": {0: (ellipsoid_sdf, [])},
    "torus": {0: (torus_sdf, [])},
    "double_torus": {0: (double_torus_sdf, [])},
    "triple_torus": {0: (triple_torus_sdf, [])},
    }



    # Visualize analytic shapes
    visualize_analytic_sdf(ellipsoid_sdf, "ellipsoid", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(torus_sdf, "torus", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(double_torus_sdf, "double_torus", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(triple_torus_sdf, "triple_torus", EXPERIMENT_ROOT, grid_res=128)


    # Train DeepSDF model
    model = Model.Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="NonTrivialInterpolation",
        scenes=scenes,
        latent_dim=4,
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

    # Precompute voxel grid
    GRID_RES = 64

    xs = np.linspace(-1,1,GRID_RES)
    ys = np.linspace(-1,1,GRID_RES)
    zs = np.linspace(-1,1,GRID_RES)

    grid = np.stack(np.meshgrid(xs,ys,zs,indexing="ij"), -1)
    grid_pts = torch.tensor(grid.reshape(-1,3)).float()

    latent_dict = {}
    canonical_voxels = {}


    def gaussian_weighted_error( latent, sdf_true, sigma, N=100000):

        pts = torch.rand(N,3,device=device)*2 - 1

        with torch.no_grad():
            f_true = sdf_true(pts).squeeze()
            f_pred = model.compute_sdf_from_latent(latent_vector=latent, xyz=pts)

        w = torch.exp(-(f_true**2)/(2*sigma**2))

        num = torch.sum(w * (f_true - f_pred)**2)
        den = torch.sum(w)

        return (num/den).item()


    sigma = (2.0) / 3.0  # domain-scale rule


    analytic_sdfs = {
    "ellipsoid": ellipsoid_sdf,
    "torus": torus_sdf,
    "double_torus": double_torus_sdf,
    "triple_torus": triple_torus_sdf
    }

    sdf_errors ={}

    for name, scene in model.trained_scenes.items():

        z = scene.latent_vector.detach()

        short_name = name.split(f"{model.model_name.lower()}_")[-1]
        latent_dict[short_name] = z

        error = gaussian_weighted_error(z, analytic_sdfs[short_name], sigma=sigma)
        sdf_errors[short_name]= error
        


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

    mean_sdf_error = np.mean(list(sdf_errors.values()))

    print(f"Mean Gaussian weighted SDF error: {mean_sdf_error:.6f}")

    # Jaccard similarity
    def topo_sim(A,B):

        inter = np.logical_and(A,B).sum()
        union = np.logical_or(A,B).sum()

        if union == 0:
            return 0

        return inter/union

    def leakage(latent, intended, model, grid_pts, GRID_RES):

        vox = model.sdf_voxel(
            latent=latent,
            grid_pts=grid_pts,
            GRID_RES=GRID_RES
        )

        sims = {
            k: topo_sim(vox, v)
            for k, v in canonical_voxels.items()
        }

        # only compare against valid voxel classes
        intended_valid = [k for k in intended if k in sims]
        unintended = [k for k in sims if k not in intended_valid]

        if len(intended_valid) == 0 or len(unintended) == 0:
            return 0.0

        intended_max = max(sims[k] for k in intended_valid)
        unintended_max = max(sims[k] for k in unintended)

        return max(0.0, unintended_max - intended_max)


    # Interpolation experiment
    #ellipsoid to torus 
    #ellipsoid to double torus
    #elllipsoid to triple torus 
    #torus to double torus 
    #double torus to triple torus
    #superposition (in latent vectors averaged together basically/ centroid of the 4 latent corners)
    
    # Segments
    segments = [
        ("ellipsoid", "torus", "Ellipsoid → Torus"),
        ("ellipsoid", "double_torus", "Ellipsoid → Double Torus"),
        ("ellipsoid", "triple_torus", "Ellipsoid → Triple Torus"),
        ("torus", "double_torus", "Torus → Double Torus"),
        ("double_torus", "triple_torus", "Double torus → Triple Torus"),
    ]

    centroid_latent = torch.stack(list(latent_dict.values()), dim=0).mean(dim=0)
    segments.append(("centroid", "centroid", "Latent Superposition (Centroid)"))

    def get_latent(name):
        return centroid_latent if name == "centroid" else latent_dict[name]

    # Sampling
    N_SAMPLES = 21  # Simpson requirement (odd)
    ts = torch.linspace(0, 1, N_SAMPLES)

    leakage_log = {}          # label -> curve
    total_leakage_log = {}    # label -> scalar

    views = [
        (-45, 15),
        (-30, 90),
        (30, 210),
    ]

    view_frames = {i: [] for i in range(len(views))}
    combined_frames = []

    # Run interpolation
    for a, b, label in segments:

        curve = []

        z_a = get_latent(a)
        z_b = get_latent(b)

        for step, t in enumerate(ts):

            name = f"interp_{a}_{b}_{step:02d}"

            # centroid is constant trajectory
            if a == "centroid" and b == "centroid":
                z = centroid_latent
            else:
                z = (1 - t) * z_a + t * z_b

            leak = leakage(
                latent=z,
                intended={a, b},
                model=model,
                grid_pts=grid_pts,
                GRID_RES=GRID_RES
            )

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

                draw.text((10, 10), label, fill=(0,0,0), font=font)
                draw.text((10, 35), f"t={float(t):.3f}", fill=(0,0,0), font=font)
                draw.text((10, 60), f"Leak={leak:.4f}", fill=(0,0,0), font=font)

                frame_path = os.path.join(
                    PLOT_DIR,
                    f"{name}_view{view_id}.png"
                )

                img.save(frame_path)

                # store frame path for this specific view
                view_frames[view_id].append(frame_path)

                # keep PIL image for stitching
                step_imgs.append(img)

            # ---- build stitched frame for this step ----

            widths = [im.width for im in step_imgs]
            heights = [im.height for im in step_imgs]

            stitched = Image.new("RGB", (sum(widths), max(heights)))

            x_offset = 0
            for im in step_imgs:
                stitched.paste(im, (x_offset, 0))
                x_offset += im.width

            stitched_path = os.path.join(
                PLOT_DIR,
                f"{name}_stitched.png"
            )

            stitched.save(stitched_path)
            combined_frames.append(stitched_path)

        

        # Simpson integration
        ys = np.asarray(curve, dtype=np.float64)

        if len(ys) < 3:
            total = float(np.sum(ys))
        else:
            if len(ys) % 2 == 0:
                ys = ys[:-1]

            h = 1.0 / (len(ys) - 1)

            total = (h / 3.0) * (
                ys[0]
                + ys[-1]
                + 4.0 * np.sum(ys[1:-1:2])
                + 2.0 * np.sum(ys[2:-2:2])
            )

        leakage_log[label] = curve
        total_leakage_log[label] = total

        print(f"{label:<40} | total leakage = {total:.6f}")



    # Create per-view GIFs
    for view_id, frames in view_frames.items():

        gif_path = os.path.join(
            PLOT_DIR,
            f"view_{view_id}_interpolation.gif"
        )

        imgs = [np.asarray(imageio.imread(f)) for f in frames]

        imageio.mimsave(  # type: ignore[call-arg]
            gif_path,
            imgs,
            duration=0.12
        )

        print(f"[INFO] Saved {gif_path}")

    # Create stitched GIF

    stitched_imgs = [np.asarray(imageio.imread(f)) for f in combined_frames]

    stitched_gif = os.path.join(
        PLOT_DIR,
        "combined_views.gif"
    )

    imageio.mimsave(  # type: ignore[call-arg]
        stitched_gif,
        stitched_imgs,
        duration=0.12
    )

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

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()