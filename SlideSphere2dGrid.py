import os
import numpy as np
import torch
import matplotlib
import imageio.v2 as imageio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

from Model import Model


# Utility logging

def log(msg):
    print(f"[LOG] {msg}")

# Analytic SDF
def analytic_sphere_sdf(xyz, cx, cy, r):

    x = xyz[:,0] - cx
    y = xyz[:,1] - cy
    z = xyz[:,2]

    return torch.sqrt(x**2 + y**2 + z**2) - r


# Scene factory
def make_scene(cx, cy, r):

    def sdf_fn(xyz, params=None):
        return analytic_sphere_sdf(xyz, cx, cy, r).unsqueeze(1)

    return sdf_fn


# Gaussian weighted reconstruction error
def gaussian_weighted_error(f_pred, f_true, sigma, N=60000):

        x = np.random.uniform(-1.5, 1.5, N)
        y = np.random.uniform(-1.5, 1.5, N)
        z = np.random.uniform(-1.5, 1.5, N)

        pts = np.stack([x, y, z], axis=-1)

        ft = f_true(pts)
        fp = f_pred(pts)

        w = np.exp(-(ft**2) / (2 * sigma**2))

        return np.sum(w * (ft - fp)**2) / np.sum(w)


# Simpson weights
def simpson_weights(N):

    w = np.ones(N)

    for k in range(1,N-1):
        if k % 2 == 1:
            w[k] = 4
        else:
            w[k] = 2

    return w


# Main experiment
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device = {device}")

    REPO_ROOT = "/content"
    EXPERIMENT_NAME = "SlidingSphere2D_latent2D"

    EXP_ROOT = os.path.join(REPO_ROOT,"experiments",EXPERIMENT_NAME)
    FRAME_DIR = os.path.join(EXP_ROOT,"frames")
    PLOT_DIR = os.path.join(EXP_ROOT,"plots")

    os.makedirs(FRAME_DIR,exist_ok=True)
    os.makedirs(PLOT_DIR,exist_ok=True)

    log(f"experiment root = {EXP_ROOT}")


    # Domain definition
    L = 1.6
    r = 0.5

    log("domain initialized")
    log(f"domain size = {L}")
    log(f"sphere radius = {r}")

    # Scene grid
    n = 3

    x_nodes = np.linspace(-L/2, L/2, n)
    y_nodes = np.linspace(-L/2, L/2, n)

    scenes = {}

    for i,x in enumerate(x_nodes):
        for j,y in enumerate(y_nodes):

            scenes[f"sphere_{i}_{j}"] = {
                0 : (make_scene(x,y,r), [])
            }

    log(f"scene grid created ({n} x {n})")
    log(f"total scenes = {len(scenes)}")

    # Model
    model_name = "SlidingSphereModel2D_latent2D"

    model = Model(
        base_directory = EXP_ROOT,
        model_name = model_name,
        scenes = scenes,
        latent_dim = 2,
        num_epochs = 7000,
        samples_per_scene = 5000,
        training_clamp_dist = None,
        sample_clamp_dist = 2,
        skip_layer = 4,
        regularize_latent = True,
        train_until_convergence = True,
        soft_latent = False,
        patience = 1,
        min_delta = 0.05
    )

    log("model initialized")

    # Grid for training visualization
    grid_res = 128

    xv = np.linspace(-1.8,1.8,grid_res)
    yv = np.linspace(-1.8,1.8,grid_res)
    zv = np.linspace(-1.8,1.8,grid_res)

    xx,yy,zz = np.meshgrid(xv,yv,zv)

    xyz_np = np.stack([xx,yy,zz],axis=-1)
    xyz = torch.tensor(xyz_np,dtype=torch.float32).reshape(-1,3).to(device)


    log("starting training")
    model.train(grid=xyz)
    log("training complete")


    # Extract latent codes
    latent_grid = np.zeros((n,n,2))

    for i in range(n):
        for j in range(n):

            key = f"{model_name.lower()}_sphere_{i}_{j}"

            latent_grid[i,j] = (
                model.trained_scenes[key]
                .get_latent_vector()
                .cpu()
                .numpy()
            )

    latents = latent_grid.reshape(-1,2)

    x_true = np.repeat(x_nodes,n)
    y_true = np.tile(y_nodes,n)

    lx = latents[:,0]
    ly = latents[:,1]

    log("latent vectors extracted")

    print("\n[METRIC] latent correlations")
    print("corr(x , z0) =",np.corrcoef(x_true,lx)[0,1])
    print("corr(y , z1) =",np.corrcoef(y_true,ly)[0,1])

    # Custom colormaps
    x_cmap = mcolors.LinearSegmentedColormap.from_list(
        "xmap",
        ["blue","yellow","red"]
    )

    y_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ymap",
        ["purple","green","orange"]
    )

    # Latent visualization
    log("generating latent visualizations")

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sc = plt.scatter(lx,ly,c=x_true,cmap=x_cmap,s=60)
    plt.colorbar(sc,label="true x")
    plt.title("latent space (x colored)")
    plt.xlabel("z0")
    plt.ylabel("z1")

    plt.subplot(1,2,2)
    sc = plt.scatter(lx,ly,c=y_true,cmap=y_cmap,s=60)
    plt.colorbar(sc,label="true y")
    plt.title("latent space (y colored)")
    plt.xlabel("z0")
    plt.ylabel("z1")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"latent_map.png"),dpi=200)
    plt.close()

    # Latent manifold grid
    plt.figure(figsize=(6,6))

    for i in range(n):
        plt.plot(latent_grid[i,:,0],latent_grid[i,:,1],'k-')

    for j in range(n):
        plt.plot(latent_grid[:,j,0],latent_grid[:,j,1],'k-')

    plt.scatter(lx,ly,c=x_true,cmap=x_cmap,s=50)

    plt.title("latent manifold grid")
    plt.xlabel("z0")
    plt.ylabel("z1")

    plt.savefig(os.path.join(PLOT_DIR,"latent_grid.png"),dpi=200)
    plt.close()

    # Corner latents
    z00 = torch.tensor(latent_grid[0,0]).to(dtype=torch.float32,device=device)
    z10 = torch.tensor(latent_grid[n-1,0]).to(dtype=torch.float32,device=device)
    z01 = torch.tensor(latent_grid[0,n-1]).to(dtype=torch.float32,device=device)
    z11 = torch.tensor(latent_grid[n-1,n-1]).to(dtype=torch.float32,device=device)

    log("corner latents extracted")

    # SDF evaluation grid
    grid_res = 256

    xv = np.linspace(-1.8,1.8,grid_res)
    yv = np.linspace(-1.8,1.8,grid_res)

    xx,yy = np.meshgrid(xv,yv)

    xyz_np = np.stack([xx,yy,np.zeros_like(xx)],axis=-1)
    xyz = torch.tensor(xyz_np,dtype=torch.float32).reshape(-1,3).to(device)


    def eval_sdf(latent):

        sdf = model.compute_sdf_from_latent(latent_vector=latent,xyz=xyz)

        if sdf.dim()==2:
            sdf=sdf[:,0]

        return sdf.cpu().numpy().reshape(grid_res,grid_res)

    # Visualization colormap
    cdict = {
        "red":[(0,0,0),(0.5,1,1),(1,1,1)],
        "green":[(0,0,0),(0.5,1,1),(1,0,0)],
        "blue":[(0,1,1),(0.5,1,1),(1,0,0)]
    }

    cmap = mcolors.LinearSegmentedColormap("sdf",cdict)

    norm = TwoSlopeNorm(vmin=-0.6,vcenter=0,vmax=0.6)

    # Latent interpolation surface
    log("evaluating latent interpolation surface")

    N = 20

    t_vals = np.linspace(0,1,N)
    u_vals = np.linspace(0,1,N)

    sigma = (3.0) / 3.0  # domain-scale rule 

    E = np.zeros((N,N))

    frames_sdf = []
    frames_error = []


    for i,t in enumerate(t_vals):
        for j,u in enumerate(u_vals):

            z = (
                (1-t)*(1-u)*z00 +
                t*(1-u)*z10 +
                (1-t)*u*z01 +
                t*u*z11
            )

            cx = (1-t)*(-L/2) + t*(L/2)
            cy = (1-u)*(-L/2) + u*(L/2)


            def f_pred(x):

                xt = torch.from_numpy(x).float().to(device)

                return model.compute_sdf_from_latent(
                    latent_vector=z,
                    xyz=xt
                ).detach().cpu().numpy().squeeze()


            def f_true(x):

                return np.sqrt((x[:,0]-cx)**2 + (x[:,1]-cy)**2 + x[:,2]**2) - r


            E[i,j] = gaussian_weighted_error(f_pred,f_true,sigma)

            sdf_img = eval_sdf(z)

            true_img = (
                analytic_sphere_sdf(xyz,cx,cy,r)
                .cpu()
                .numpy()
                .reshape(grid_res,grid_res)
            )

            error_img = sdf_img - true_img


            # SDF frame
            fig,ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(sdf_img,cmap=cmap,norm=norm,origin="lower")
            fig.colorbar(im,ax=ax)
            ax.set_title(f"SDF t={t:.2f} u={u:.2f}")

            path = os.path.join(FRAME_DIR,f"sdf_{i}_{j}.png")
            plt.savefig(path)
            plt.close()

            frames_sdf.append(imageio.imread(path))


            # Error frame
            fig,ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(error_img,cmap=cmap,norm=norm,origin="lower")
            fig.colorbar(im,ax=ax)
            ax.set_title(f"Error t={t:.2f} u={u:.2f}")

            path = os.path.join(FRAME_DIR,f"err_{i}_{j}.png")
            plt.savefig(path)
            plt.close()

            frames_error.append(imageio.imread(path))


    log("creating GIFs")
    imageio.mimsave(
        os.path.join(EXP_ROOT,"sdf_surface.gif"),
        frames_sdf,
        duration=0.04
    )

    imageio.mimsave(
        os.path.join(EXP_ROOT,"sdf_error.gif"),
        frames_error,
        duration=0.04
    )

    # Simpson integration
    wt = simpson_weights(N)
    wu = simpson_weights(N)

    dt = 1/(N-1)
    du = 1/(N-1)

    integral = 0

    for i in range(N):
        for j in range(N):
            integral += wt[i]*wu[j]*E[i,j]

    integral *= (dt*du)/9

    print("\n[METRIC] integrated reconstruction error =",integral)
    reconstruction_error =0
    for i in range (N):
        for j in range(N):
            reconstruction_error+=E[i,j]
    
    mean_reconstruction_error = reconstruction_error/(N**2)
    print("\n[METRIC] integrated reconstruction error =",mean_reconstruction_error)


    log("experiment complete")


if __name__ == "__main__":
    main()