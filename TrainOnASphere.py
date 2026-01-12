import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from DeepSDFStruct.sdf_primitives import SphereSDF
import Model

#to verify our architecture formulation replicates the paper exactly
# we train on a single sphere and verify the latent is stablely learned
# and that the shape is reconstructed perfectly
# We can see that this is the case 

# ======================================================
# Experiment Setup
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)

EXPERIMENT_NAME = "TrainOnASphere"
EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "plots"), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_ROOT, "Meshes"), exist_ok=True)

print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")

# ======================================================
# Scene: A Single Sphere
# ======================================================
def single_sphere_sdf():
    """Return SDF function for a sphere at the origin, radius 0.4."""
    def sdf_fn(xyz, params=None):
        return SphereSDF(
            center=torch.tensor([0.0, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device),
            radius=0.4
        )._compute(xyz)
    return sdf_fn

scenes = {
    "sphere": {
        0: (single_sphere_sdf(), [])
    }
}

print("[INFO] Single sphere SDF scene created.")

# ======================================================
# Initialize Model (CPU-safe)
# ======================================================
model = Model.Model(
    base_directory=EXPERIMENT_ROOT,
    model_name="TrainOnASphereModel",
    scenes=scenes,
    latent_dim=1,
    num_epochs=1000,
    domain_radius=0.45,
    regularize_latent=True,
    training_clamp_dist=0.1,
    samples_per_scene=10000,
    device="cpu",  # Force CPU to avoid CUDA errors
)

print("[INFO] Model initialized. Starting training...")

# ======================================================
# Training & Visualization
# ======================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Train the model
    model.train()
    print("[INFO] Training complete.")

    # ======================================================
    # Extract latent for the single scene
    # ======================================================
    if len(model.trained_scenes) != 1:
        raise RuntimeError("Expected exactly one trained scene.")

    scene_key = list(model.trained_scenes.keys())[0]
    latent = model.trained_scenes[scene_key].get_latent_vector().detach().cpu()
    print(f"[INFO] Latent for {scene_key}: {latent}")

    # ======================================================
    # Visualize shape using VisualizeAShape
    # ======================================================
    print("[INFO] Visualizing sphere from trained latent...")

  # Assuming you have a Model instance called `my_model` 
# and a trained scene registered as "trainonaspheremodel_0" (or whatever key you used)

    meshes = model.visualize_a_shape(
        latent=latent,  # must match the key in trained_scenes
        grid_res=128,
        clamp_dist=0.1,
    )


    if not meshes:
        print("[WARN] No mesh produced by VisualizeAShape.")
        print("[INFO] Done.")
        quit()

    mesh = meshes[0]
    mesh_dir = os.path.join(EXPERIMENT_ROOT, "Meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh.export(os.path.join(mesh_dir, "trained_sphere_mesh.ply"))
    print(f"[INFO] Exported trained sphere mesh to {mesh_dir}/trained_sphere_mesh.ply")

   