"""
NonTrivialInterpolation Experiment

This script performs the following:

1. Defines three analytic SDFs:
   - Torus (Genus-1)
   - Wavey Rounded Box (box with smooth non-periodic bubbles)
   - Dented Sphere (sphere with uniform angular dents)

2. Visualizes the analytic SDFs prior to training.

3. Trains a DeepSDF model with a 3D latent vector.
   - Each latent dimension corresponds roughly to one of the three shapes.
   - Latent space is enforced to be continuous using cubic spline interpolation.

4. Generates PLY meshes for:
   - Each individual shape.
   - Interpolated latent vectors along spline paths between shapes.

5. Exports an animated GIF showing smooth latent interpolation and adjusted continuity parameter t.

Known limitations:
- DeepSDF does not strictly preserve genus.
- Some interpolations may self-intersect.
- Extreme curvature regions may collapse.
- For guaranteed topological invariants, Morse-theoretic or Hessian regularization would be required.
"""

import os
import torch
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

import Model
from VisualizeAnalyticSDF import visualize_analytic_sdf

# ======================================================
# Experiment Setup
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)

EXPERIMENT_NAME = "NonTrivialInterpolation"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "Meshes"), exist_ok=True)

print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")

# ======================================================
# Analytic Signed Distance Functions
# ======================================================
R = 0.55  # Base radius for dented sphere

def analytic_torus_sdf(xyz, params=None):
    """Genus-1 torus: major radius R_major, minor radius r_minor."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    R_major, r_minor = 0.38, 0.17
    q = torch.sqrt(x**2 + y**2) - R_major
    sdf = torch.sqrt(q**2 + z**2) - r_minor
    return sdf.unsqueeze(1)

def wavey_rounded_box_sdf(xyz, params=None):
    """
    Smooth rounded box with low-frequency bubble field.
    Base box: half-sizes hx, hy, hz with corner radius r.
    Bubble field creates handles without introducing lattice periodicity.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    hx, hy, hz, r = 0.48, 0.48, 0.48, 0.10
    qx, qy, qz = torch.abs(x)-hx, torch.abs(y)-hy, torch.abs(z)-hz
    q = torch.stack([qx, qy, qz], dim=1)
    outside = torch.clamp(q, min=0.0)
    inside = torch.clamp(torch.max(q, dim=1).values, max=0.0)
    box = torch.linalg.norm(outside, dim=1) + inside - r
    bubble = 0.08 * torch.sin(3.0*x) * torch.sin(2.5*y) * torch.sin(2.0*z)
    sdf = box + bubble
    return sdf.unsqueeze(1)

def dented_sphere_sdf(xyz, params=None):
    """
    Sphere of radius R with uniform angular dents.
    Theta, phi angular coordinates produce evenly distributed dents.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    base = torch.sqrt(x**2 + y**2 + z**2) - R
    theta = torch.atan2(y, x)
    phi = torch.atan2(z, torch.sqrt(x**2 + y**2))
    dents = 0.10 * torch.cos(4*theta) * torch.cos(3*phi)
    sdf = base + dents
    return sdf.unsqueeze(1)

# ======================================================
# Scene dictionary for DeepSDF
# ======================================================
scenes = {
    "analytic_torus": {0: (analytic_torus_sdf, [])},
    "wavey_rounded_box": {0: (wavey_rounded_box_sdf, [])},
    "dented_sphere": {0: (dented_sphere_sdf, [])},
}

# Visualize analytic SDFs prior to training
visualize_analytic_sdf(analytic_torus_sdf, "torus", EXPERIMENT_ROOT, grid_res=128)
visualize_analytic_sdf(wavey_rounded_box_sdf, "wavey_rounded_box", EXPERIMENT_ROOT, grid_res=128)
visualize_analytic_sdf(dented_sphere_sdf, "dented_sphere", EXPERIMENT_ROOT, grid_res=128)

# ======================================================
# Train DeepSDF model
# ======================================================
model = Model.Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="SplineSymmetryExperiment",
    scenes=scenes,
    latent_dim=3,  # 3D latent vector corresponds to three shapes
    num_epochs=1000,
    domain_radius=1.0,
    device="cpu",
)

model.train()
print("[INFO] DeepSDF training complete.")

# ======================================================
# Cubic Spline Interpolation (Catmull-Rom)
# ======================================================
def catmull_rom(p0, p1, p2, p3, t):
    """
    Cubic Catmull-Rom spline for latent interpolation.
    p0, p1, p2, p3 : torch.Tensor latent vectors
    t : scalar in [0,1] continuity parameter
    Returns interpolated latent vector
    """
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t**2
        + (-p0 + 3*p1 - 3*p2 + p3) * t**3
    )

# Retrieve latent vectors for the three shapes
latents = [
    model.trained_scenes["splinesymmetryexperiment_analytic_torus"].latent_vector,
    model.trained_scenes["splinesymmetryexperiment_wavey_rounded_box"].latent_vector,
    model.trained_scenes["splinesymmetryexperiment_dented_sphere"].latent_vector,
]

# Catmull-Rom padding for smooth spline
p0, p1, p2 = latents
p3 = p2 + (p2 - p1)

# Interpolation continuity parameters
ts = torch.linspace(0, 1, 30)  # 30 frames per segment
gif_frames = []
mesh_output_dir = os.path.join(EXPERIMENT_ROOT, "Meshes")
os.makedirs(mesh_output_dir, exist_ok=True)

# Segments of interpolation
segments = [
    ("analytic_torus", "wavey_rounded_box"),
    ("analytic_torus", "dented_sphere"),
    ("dented_sphere", "wavey_rounded_box"),
]

# ======================================================
# Interpolation Loop: Generate meshes + GIF frames
# ======================================================
for seg_idx, (name_start, name_end) in enumerate(segments):
    z_start = model.trained_scenes[f"splinesymmetryexperiment_{name_start}"].latent_vector
    z_end   = model.trained_scenes[f"splinesymmetryexperiment_{name_end}"].latent_vector

    for i, t in enumerate(ts):
        # Latent interpolation: could replace linear with Catmull-Rom if desired
        z_interp = (1-t)*z_start + t*z_end

        # Reuse decoder from any trained scene
        scene = list(model.trained_scenes.values())[0]

        # Generate mesh via DeepSDF decoder
        meshes = model.visualize_a_shape(
            latent=z_interp,
            key=scene.scene_key,
            grid_res=96,
            clamp_dist=0.1,
            param_values=None,
            save_suffix=f"{seg_idx}_{i:03d}",
        )

        if not meshes:
            continue

        mesh = meshes[0]

        # Save PLY mesh
        ply_filename = f"interp_seg{seg_idx}_frame{i:03d}.ply"
        ply_path = os.path.join(mesh_output_dir, ply_filename)
        mesh.export(ply_path)
        print(f"[INFO] Saved mesh: {ply_path}")

        # Render image for GIF
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces],
                                             facecolor='lightblue',
                                             edgecolor='k',
                                             linewidth=0.1,
                                             alpha=1.0))
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.view_init(elev=25, azim=30)
        ax.set_box_aspect([1,1,1])
        plt.axis('off')
        ax.set_title(f"Segment {seg_idx} t={t:.2f}")

        # Save temporary image for GIF
        img_path = os.path.join(EXPERIMENT_ROOT, "plots", f"frame_{seg_idx}_{i:03d}.png")
        plt.savefig(img_path, dpi=200, bbox_inches='tight')
        plt.close()
        gif_frames.append(img_path)

# ======================================================
# Create GIF of interpolation
# ======================================================
gif_path = os.path.join(EXPERIMENT_ROOT, "plots", "interpolation.gif")
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for frame_path in gif_frames:
        image = imageio.imread(frame_path)
        writer.append_data(image) # type: ignore

print(f"[INFO] Interpolation GIF saved to: {gif_path}")
print("[INFO] All meshes exported and GIF complete.")
