import Model
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import torch
import io

def visualize_analytic_sdf(
    sdf_fn,
    name: str,
    out_root: str,
    grid_res: int = 128,
    grid_center=(0.0, 0.0, 0.0),
    bbox_half: float = 1.0,
    clamp_dist: float = 0.1,
):
    """
    Visualize an analytic SDF (no network).
    Exports mesh + static render.
    """

    print(f"[ANALYTIC] Visualizing {name}")

    # Build grid
    x = np.linspace(grid_center[0] - bbox_half, grid_center[0] + bbox_half, grid_res)
    y = np.linspace(grid_center[1] - bbox_half, grid_center[1] + bbox_half, grid_res)
    z = np.linspace(grid_center[2] - bbox_half, grid_center[2] + bbox_half, grid_res)

    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    pts = torch.from_numpy(grid.reshape(-1, 3)).float()

    # Evaluate SDF
    with torch.no_grad():
        sdf = sdf_fn(pts).squeeze(1).cpu().numpy()

    volume = sdf.reshape(grid_res, grid_res, grid_res)

    if not (volume.min() < 0 < volume.max()):
        print(f"[WARN] {name}: no zero-crossing found, skipping")
        return

    # Marching cubes
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0.0)
    scale = x[1] - x[0]
    verts = verts * scale + np.array([x[0], y[0], z[0]])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    mesh_dir = os.path.join(out_root, "Meshes")
    plot_dir = os.path.join(out_root, "plots")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # --- save mesh ---
    mesh_path = os.path.join(mesh_dir, f"analytic_{name}.ply")
    mesh.export(mesh_path)

    print(f"[ANALYTIC] Mesh saved → {mesh_path}")

    # --- fast render scene  ---
    img = Model.render_mesh_isometric_pil(mesh)
    render_path = os.path.join(plot_dir, f"analytic_{name}.png")
    img.save(render_path)
    print(f"[ANALYTIC] Render saved → {render_path}")

