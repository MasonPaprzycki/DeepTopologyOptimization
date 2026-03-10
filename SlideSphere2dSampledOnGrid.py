# ======================================================
# Install dependencies (safe if already installed)
# ======================================================
!pip install trimesh imageio scipy scikit-learn

# ======================================================
# Imports
# ======================================================
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

# ======================================================
# Experiment Setup
# ======================================================

EXPERIMENT_NAME = "SlidingSphere2D_latent2D"
REPO_ROOT = "/content"
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

print("[INFO] Experiment directory:", EXPERIMENT_ROOT)

# ======================================================
# Analytic Sphere SDF
# ======================================================

def analytic_sphere_sdf(xyz, params=None):

    cx = 0.0 if params is None else params.get("cx",0.0)
    cy = 0.0 if params is None else params.get("cy",0.0)
    cz = 0.0 if params is None else params.get("cz",0.0)

    x = xyz[:,0] - cx
    y = xyz[:,1] - cy
    z = xyz[:,2] - cz

    radius = 0.5

    return (torch.sqrt(x**2 + y**2 + z**2) - radius).unsqueeze(1)

# ======================================================
# Generate scenes
# ======================================================

def make_sphere_scene(cx,cy):

    def sdf_fn(xyz, params=None):
        return analytic_sphere_sdf(xyz, params={"cx":cx,"cy":cy})

    return sdf_fn


num_scenes_per_axis = 5

x_positions = np.linspace(-0.8,0.8,num_scenes_per_axis)
y_positions = np.linspace(-0.8,0.8,num_scenes_per_axis)

scenes = {}

for i,x in enumerate(x_positions):
    for j,y in enumerate(y_positions):

        key = f"sphere_{i}_{j}"

        scenes[key] = {0:(make_sphere_scene(x,y),[])}

print("[INFO] Created",len(scenes),"scenes")

# ======================================================
# Train Model
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "SlidingSphereModel2D_latent2D"

model = Model(
    base_directory=EXPERIMENT_ROOT,
    model_name=model_name,
    scenes=scenes,
    scenes_per_batch=1,
    latent_dim=2,
    num_epochs=7000,
    samples_per_scene=5000,
    training_clamp_dist=None,
    sample_clamp_dist=2,
    skip_layer=4,
    regularize_latent=True,
    soft_latent=False
)

# ======================================================
# Build training grid
# ======================================================

grid_res = 128

xv = np.linspace(-1.8,1.8,grid_res)
yv = np.linspace(-1.8,1.8,grid_res)

xx,yy = np.meshgrid(xv,yv)

xyz_np = np.stack([xx,yy,np.zeros_like(xx)],axis=-1)

xyz = torch.tensor(xyz_np,dtype=torch.float32).reshape(-1,3).to(device)

print("[INFO] Training model")
model.train(grid=xyz)
print("[INFO] Training complete")

# ======================================================
# Collect Latents
# ======================================================

latent_grid = np.zeros((num_scenes_per_axis,num_scenes_per_axis,2))

for i in range(num_scenes_per_axis):
    for j in range(num_scenes_per_axis):

        key = f"{model_name.lower()}_sphere_{i}_{j}"

        latent = model.trained_scenes[key].get_latent_vector().cpu().numpy()

        latent_grid[i,j] = latent

latents = latent_grid.reshape(-1,2)

x_positions_np = np.tile(x_positions,num_scenes_per_axis)
y_positions_np = np.repeat(y_positions,num_scenes_per_axis)

latent_x = latents[:,0]
latent_y = latents[:,1]

# ======================================================
# Correlations
# ======================================================

corr_x = np.corrcoef(x_positions_np,latent_x)[0,1]
corr_y = np.corrcoef(y_positions_np,latent_y)[0,1]

print("[INFO] X correlation:",corr_x)
print("[INFO] Y correlation:",corr_y)

# ======================================================
# Plot latent vs true
# ======================================================

fig,axes = plt.subplots(1,2,figsize=(12,6))

x_norm = (x_positions_np-x_positions_np.min())/(x_positions_np.max()-x_positions_np.min())
y_norm = (y_positions_np-y_positions_np.min())/(y_positions_np.max()-y_positions_np.min())

colors = np.stack([x_norm,y_norm,np.full_like(x_norm,0.5)],axis=1)

axes[0].scatter(latent_x,latent_y,c=colors,s=40)
axes[0].set_title("Latent Space")
axes[0].set_xlabel("latent 0")
axes[0].set_ylabel("latent 1")
axes[0].set_aspect("equal")

axes[1].scatter(x_positions_np,y_positions_np,c=colors,s=40)
axes[1].set_title("True Sphere Positions")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")

plt.tight_layout()

plt.savefig(os.path.join(PLOT_DIR,"latent_vs_true.png"),dpi=200)

plt.close()

# ======================================================
# Latent Min Max
# ======================================================

latents_tensor = torch.stack(
    [scene.get_latent_vector() for scene in model.trained_scenes.values()]
).float()

latent_min = latents_tensor.min(dim=0).values
latent_max = latents_tensor.max(dim=0).values

print("[INFO] Latent min/max",latent_min,latent_max)

#build visualization grid 
# ======================================================
# Build SDF grid
# ======================================================

grid_res = 256

xv = np.linspace(-1.8,1.8,grid_res)
yv = np.linspace(-1.8,1.8,grid_res)

xx,yy = np.meshgrid(xv,yv)

xyz_np = np.stack([xx,yy,np.zeros_like(xx)],axis=-1)

xyz = torch.tensor(xyz_np,dtype=torch.float32).reshape(-1,3).to(device)


def eval_sdf(latent_vec):

    sdf = model.compute_sdf_from_latent(
        latent_vector=latent_vec,
        xyz=xyz
    )

    if sdf.dim()==2:
        sdf = sdf[:,0]

    return sdf.cpu().numpy().reshape(grid_res,grid_res)

# ======================================================
# Latent sweep GIF
# ======================================================

interp_steps = 25
frames = []

tx_vals = torch.linspace(0,1,interp_steps)
ty_vals = torch.linspace(0,1,interp_steps)

corners = {

"bottom_left":model.trained_scenes[f"{model_name.lower()}_sphere_0_0"].get_latent_vector(),
"bottom_right":model.trained_scenes[f"{model_name.lower()}_sphere_{num_scenes_per_axis-1}_0"].get_latent_vector(),
"top_left":model.trained_scenes[f"{model_name.lower()}_sphere_0_{num_scenes_per_axis-1}"].get_latent_vector(),
"top_right":model.trained_scenes[f"{model_name.lower()}_sphere_{num_scenes_per_axis-1}_{num_scenes_per_axis-1}"].get_latent_vector()

}

# ======================================================
# Colormap (same as 1D experiment)
# ======================================================

cdict = {
    "red":   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
    "green": [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
    "blue":  [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
}

custom_cmap = mcolors.LinearSegmentedColormap("sdf_custom", cdict)

norm = mcolors.TwoSlopeNorm(
    vmin=-0.6,
    vcenter=0.0,
    vmax=0.6
)
for i,tx in enumerate(tx_vals):
    for j,ty in enumerate(ty_vals):

        latent_vec = (
            (1-tx)*(1-ty)*corners["bottom_left"]
            + tx*(1-ty)*corners["bottom_right"]
            + (1-tx)*ty*corners["top_left"]
            + tx*ty*corners["top_right"]
        )

        sdf_img = eval_sdf(latent_vec)

        plt.figure(figsize=(5,5))

        plt.imshow(
            sdf_img,
            extent=(-1.8,1.8,-1.8,1.8),
            origin="lower",
            cmap= custom_cmap,
            norm=norm
        )

        plt.colorbar()

        frame_path = os.path.join(FRAME_DIR,f"latent_{i}_{j}.png")

        plt.savefig(frame_path)

        plt.close()

        frames.append(imageio.imread(frame_path))

gif_path = os.path.join(EXPERIMENT_ROOT,"latent2D_superposition.gif")

imageio.mimsave(gif_path,frames,duration=0.05)

print("[INFO] GIF saved:",gif_path)