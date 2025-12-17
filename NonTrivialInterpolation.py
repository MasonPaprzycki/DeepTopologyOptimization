import torch 
import os
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

R=0.55
def analytic_torus_sdf(xyz, params=None):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    R_major = 0.38
    r_minor = 0.17

    q = torch.sqrt(x**2 + y**2) - R_major
    sdf = torch.sqrt(q**2 + z**2) - r_minor

    return sdf.unsqueeze(1)

#Genus-1
#No oscillations
#should be good baseline for interpolation

def wavey_rounded_box_sdf(xyz, params=None):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Base box
    hx, hy, hz = 0.48, 0.48, 0.48
    r = 0.10

    qx = torch.abs(x) - hx
    qy = torch.abs(y) - hy
    qz = torch.abs(z) - hz

    q = torch.stack([qx, qy, qz], dim=1)
    outside = torch.clamp(q, min=0.0)
    inside = torch.clamp(torch.max(q, dim=1).values, max=0.0)

    box = torch.linalg.norm(outside, dim=1) + inside - r

    # Bubble field (low-frequency, non-periodic)
    bubble = (
        0.08 * torch.sin(3.0 * x)
        * torch.sin(2.5 * y)
        * torch.sin(2.0 * z)
    )

    sdf = box + bubble

    return sdf.unsqueeze(1)
#Strong curvature variation
#should interpolate beautifully into torus handles

def dented_sphere_sdf(xyz, params=None):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Base sphere
    base = torch.sqrt(x**2 + y**2 + z**2) - R

    # Angular coordinates
    theta = torch.atan2(y, x)
    phi = torch.atan2(z, torch.sqrt(x**2 + y**2))

    # Uniform dent field on the sphere
    dents = (
        0.10
        * torch.cos(4 * theta)
        * torch.cos(3 * phi)
    )

    sdf = base + dents

    return sdf.unsqueeze(1)

#Dents are angular → uniform over surface
#Interpolation causes dents to stretch → merge → open handles

scenes = {
    "analytic_torus": {
        0: (analytic_torus_sdf, [])
    },
    "wavey_rounded_box": {
        0: (wavey_rounded_box_sdf, [])
    },
    "dented_sphere": {
        0: (dented_sphere_sdf, [])
    }
}

visualize_analytic_sdf(
    analytic_torus_sdf,
    "torus",
    EXPERIMENT_ROOT,
    grid_res=128,
)

visualize_analytic_sdf(
    wavey_rounded_box_sdf,
    "wavey_rounded_box",
    EXPERIMENT_ROOT,
    grid_res=128,
)

visualize_analytic_sdf(
    dented_sphere_sdf,
    "dented_sphere",
    EXPERIMENT_ROOT,
    grid_res=128,
)

model = Model.Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="SplineSymmetryExperiment",
    scenes=scenes,
    latent_dim=3,              # ← critical
    num_epochs=1000,
    domain_radius=1.0,
    device="cpu",
)

model.train()

# after training we have z_torus, z_blob, z_gyroid ∈ ℝ³

# We induce a cubic spline (Catmull–Rom for simplicity)
def catmull_rom(p0, p1, p2, p3, t):
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t**2
        + (-p0 + 3*p1 - 3*p2 + p3) * t**3
    )

latents = [
    model.trained_scenes["splinesymmetryexperiment_twisted_torus"].latent_vector,
    model.trained_scenes["splinesymmetryexperiment_multi_handle_blob"].latent_vector,
    model.trained_scenes["splinesymmetryexperiment_warped_gyroid"].latent_vector,
]

# Pad endpoints
p0, p1, p2 = latents
p3 = p2 + (p2 - p1)

ts = torch.linspace(0, 1, 10)

for i, t in enumerate(ts):
    z = catmull_rom(p0, p1, p2, p3, t).detach()

    scene = list(model.trained_scenes.values())[0]  # reuse decoder
    meshes = model.visualize_a_shape(
        latent=z,
        key=scene.scene_key,
        grid_res=96,
        clamp_dist=0.1,
        param_values=None,
        save_suffix=f"spline_{i:02d}",
    )


#notes/ what we should expect 
#Smooth topology transitions

#Handle twisting and redistribution

#Latent arithmetic has geometric meaning

#Known limitations:

#DeepSDF does not preserve genus strictly
#Some interpolations may self-intersect
#Extreme curvature regions may collapse

#For guaranteed topological invariants :

#Morse-theoretic regularization
#Or signed distance Hessian penalties
#Or neural implicit topology control (not trivial)