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
    num_epochs=100,
    domain_radius=0.45,
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
        key="trainonaspheremodel_0",
        latent=latent,  # must match the key in trained_scenes
        grid_res=96,
        clamp_dist=0.1,
        save_suffix="single",
        grid_center=(0.0, 0.0, 0.0),
    )


    if not meshes:
        print("[WARN] No mesh produced by VisualizeAShape.")
        print("[INFO] Done.")
        quit()

    # Locate the exported mesh
    mesh_dir = os.path.join(EXPERIMENT_ROOT, "Meshes")
    all_mesh_files = [
        f for f in os.listdir(mesh_dir)
        if f.endswith(".ply") and f.startswith("trainonaspheremodel")
    ]
    if not all_mesh_files:
        raise FileNotFoundError("Mesh should have been exported by VisualizeAShape but none found.")

    all_mesh_files.sort()
    final_mesh_path = os.path.join(mesh_dir, all_mesh_files[-1])
    print(f"[INFO] Using exported mesh: {final_mesh_path}")

    mesh = trimesh.load(final_mesh_path)

    # ======================================================
    # Static render
    # ======================================================
    if hasattr(mesh, "geometry") and isinstance(mesh.geometry, dict):
        geom_list = [
            geom for geom in mesh.geometry.values()
            if hasattr(geom, "vertices") and hasattr(geom, "faces")
        ]
        mesh = trimesh.util.concatenate(geom_list)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(
        mesh.vertices[mesh.faces],
        facecolor='lightblue',
        edgecolor='k',
        linewidth=0.1,
        alpha=1.0
    ))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=25, azim=30)
    ax.set_box_aspect([1, 1, 1])
    plt.axis('off')

    render_path = os.path.join(EXPERIMENT_ROOT, "plots", "trained_sphere_render.png")
    plt.savefig(render_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Render saved to: {render_path}")
    print("[INFO] Done.")

