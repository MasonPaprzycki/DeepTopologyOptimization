import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
import trimesh
from PIL import Image, ImageDraw, ImageFont
import imageio
import Model
from VisualizeAnalyticSDF import visualize_analytic_sdf


def main():

    REPO_ROOT = os.getcwd()
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", "NonTrivialInterpolation")

    PLOT_DIR = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR = os.path.join(EXPERIMENT_ROOT, "meshes")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)

    ############################################################
    # Analytic SDF definitions
    ############################################################

    R = 0.55

    def analytic_torus_sdf(xyz, params=None):

        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

        R_major = 0.38
        r_minor = 0.17

        q = torch.sqrt(x**2 + y**2) - R_major

        return (torch.sqrt(q**2 + z**2) - r_minor).unsqueeze(1)


    def wavey_box_sdf(xyz, params=None):

        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

        hx, hy, hz, r = 0.48, 0.48, 0.48, 0.10

        q = torch.stack([
            torch.abs(x)-hx,
            torch.abs(y)-hy,
            torch.abs(z)-hz
        ], dim=1)

        outside = torch.clamp(q, min=0)
        inside = torch.clamp(torch.max(q, dim=1).values, max=0)

        box = torch.linalg.norm(outside, dim=1) + inside - r

        bubble = 0.08 * torch.sin(3*x) * torch.sin(2.5*y) * torch.sin(2*z)

        return (box + bubble).unsqueeze(1)


    def dented_sphere_sdf(xyz, params=None):

        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

        base = torch.sqrt(x**2 + y**2 + z**2) - R

        theta = torch.atan2(y, x)
        phi = torch.atan2(z, torch.sqrt(x**2 + y**2))

        dents = 0.10 * torch.cos(4*theta) * torch.cos(3*phi)

        return (base + dents).unsqueeze(1)


    ############################################################
    # Scene dictionary
    ############################################################

    scenes = {
        "torus": {0:(analytic_torus_sdf, [])},
        "wavey_rounded_box": {0:(wavey_box_sdf, [])},
        "dented_sphere": {0:(dented_sphere_sdf, [])},
    }


    ############################################################
    # Visualize analytic shapes
    ############################################################

    visualize_analytic_sdf(analytic_torus_sdf, "torus", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(wavey_box_sdf, "wavey_rounded_box", EXPERIMENT_ROOT, grid_res=128)
    visualize_analytic_sdf(dented_sphere_sdf, "dented_sphere", EXPERIMENT_ROOT, grid_res=128)


    ############################################################
    # Train DeepSDF model
    ############################################################

    model = Model.Model(
        base_directory=EXPERIMENT_ROOT,
        model_name="NonTrivialInterpolation",
        scenes=scenes,
        latent_dim=3,
        num_epochs=1200,
        samples_per_scene=5000,
        domain_radius=1.0,
        skip_layer=4,
        regularize_latent=False,
        soft_latent=False
    )

    model.train()


    ############################################################
    # Precompute voxel grid
    ############################################################

    GRID_RES = 48

    xs = np.linspace(-1,1,GRID_RES)
    ys = np.linspace(-1,1,GRID_RES)
    zs = np.linspace(-1,1,GRID_RES)

    grid = np.stack(np.meshgrid(xs,ys,zs,indexing="ij"), -1)
    grid_pts = torch.tensor(grid.reshape(-1,3)).float()

    ############################################################
    # Canonical latent codes
    ############################################################

    latent_dict = {}
    canonical_voxels = {}

    for k,scene in model.trained_scenes.items():

        short = "_".join(k.split("_")[1:])
        name =  f"canonical_{short}"
        z = scene.latent_vector.detach()

        latent_dict[short] = z
        canonical_voxels[short] = model.sdf_voxel(latent=z,grid_pts=grid_pts,GRID_RES=GRID_RES)
        mesh = model.reconstruct_mesh(
            latent= z,
            name=name,
            grid_pts=grid_pts,
            GRID_RES=GRID_RES
        )

        if mesh is None:
            continue

        mesh.export(os.path.join(MESH_DIR, f"{name}.ply"))
        img =Model.render_mesh_isometric_pil(mesh)
 
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), name, fill=(0, 0, 0), font=font)

        frame_path = os.path.join(EXPERIMENT_ROOT, "plots", f"{name}.png")
        img.save(frame_path)

    ############################################################
    # Jaccard similarity
    ############################################################

    def topo_sim(A,B):

        inter = np.logical_and(A,B).sum()
        union = np.logical_or(A,B).sum()

        if union == 0:
            return 0

        return inter/union


    ############################################################
    # Leakage metric
    ############################################################

    def leakage(latent, intended, model: Model.Model,grid_pts, GRID_RES):

        vox = model.sdf_voxel(latent=latent,grid_pts=grid_pts, GRID_RES=GRID_RES)

        sims = {k: topo_sim(vox,v) for k,v in canonical_voxels.items()}

        unintended = [k for k in sims if k not in intended]

        if len(unintended) == 0:
            return 0

        intended_mean = np.mean([sims[k] for k in intended])
        unintended_max = max(sims[k] for k in unintended)

        return max(0, unintended_max - intended_mean)


    ############################################################
    # Interpolation experiment
    ############################################################

    segments = [
        ("torus","wavey_rounded_box","Torus→Box"),
        ("torus","dented_sphere","Torus→Sphere"),
        ("dented_sphere","wavey_rounded_box","Sphere→Box")
    ]

    ts = torch.linspace(0,1,20)
    leakage_log = {}
    gif_frames = []

    for seg_id,(a,b,label) in enumerate(segments):

        leakage_log[seg_id] = []

        for step,t in enumerate(ts):
            name=f"interp_{a}_{b}_{step:02d}"

            z = (1-t)*latent_dict[a] + t*latent_dict[b]

            leak = leakage(
                model=model,
                latent= z,
                grid_pts=grid_pts,
                GRID_RES=GRID_RES,
                intended={a,b})
            leakage_log[seg_id].append(leak)

            mesh = model.reconstruct_mesh(
                latent=z,
                grid_pts=grid_pts,
                GRID_RES=GRID_RES,
                name=name
            )
            if mesh is None:
                    continue
            
            mesh.export(os.path.join(MESH_DIR, f"{name}.ply"))

            img = Model.render_mesh_isometric_pil(mesh)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), label, fill=(0, 0, 0), font=font)
            draw.text((10, 35), f"t={float(t):.3f}", fill=(0, 0, 0), font=font)
            draw.text((10, 60), f"Leak={leak:.4f}", fill=(0, 0, 0), font=font)

            frame_path = os.path.join(EXPERIMENT_ROOT, "plots", f"{name}.png")
            img.save(frame_path)
            gif_frames.append(frame_path)
        
    # ======================================================
    # GIF
    # ======================================================
    imageio.mimsave(
        os.path.join(EXPERIMENT_ROOT, "plots", "interpolation.gif"),
        [imageio.imread(p) for p in gif_frames],
        duration=0.08
    )
        

    ############################################################
    # Plot leakage curves
    ############################################################

    plt.figure(figsize=(8,5))

    for seg_id,(a,b,label) in enumerate(segments):

        plt.plot(ts.numpy(), leakage_log[seg_id], label=label)

    plt.xlabel("Interpolation parameter")
    plt.ylabel("Shape leakage")
    plt.title("Latent interpolation leakage")

    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(PLOT_DIR,"leakage_curves.png"))
    plt.close()


if __name__ == "__main__":
    main()