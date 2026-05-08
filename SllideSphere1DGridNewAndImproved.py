import numpy as np
import torch
import matplotlib
import trimesh
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from Model import Model
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "SlidingSphereModel1D_latent1D"

    REPO_ROOT = "/content"
    EXPERIMENT_NAME = "SlidingSphere1D_latent1D"
    EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", EXPERIMENT_NAME)

    FRAME_DIR = os.path.join(EXPERIMENT_ROOT, "frames_latents")
    PLOT_DIR = os.path.join(EXPERIMENT_ROOT, "plots")
    MESH_DIR = os.path.join(EXPERIMENT_ROOT, "Meshes")

    for d in [EXPERIMENT_ROOT, FRAME_DIR, PLOT_DIR, MESH_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Experiment directory: {EXPERIMENT_ROOT}")
    R=0.5

    def analytic_sphere_sdf(xyz, cx, cy, r):
        center = torch.tensor([cx, cy, 0.0], device=xyz.device, dtype=xyz.dtype)
        return torch.linalg.norm(xyz - center, dim=1) - r

    def make_scene(cx,cy):
        return lambda xyz, params=None: analytic_sphere_sdf(xyz, cx,cy,R)


    # Dataset
    num_scenes = 3
    positions = np.linspace(0,1,num_scenes)

    scenes = {}
    for i,x in enumerate(positions):
        scenes[f"sphere_{i}"] = {0:(make_scene(x,x**2),[])}

    print(f"[INFO] Scenes: {len(scenes)}")

    # Model
    model = Model(
        base_directory=EXPERIMENT_ROOT,
        model_name=model_name,
        scenes=scenes,
        scenes_per_batch=1,
        latent_dim=1,
        num_epochs=15000,
        training_clamp_dist=None,
        samples_per_scene=5000,
        sample_clamp_dist=0.1,
        skip_layer=4,
        regularize_latent=True,
        soft_latent=False,
        weight_norm=False,
        train_until_convergence=True,
        stochastic_distribution=False,
        patience=2,
        window=50,
        min_delta=0.05
    )


    # Training grid
    grid_res = 128

    x = np.linspace(-1,2,grid_res)
    y = np.linspace(-1.5,1.5,grid_res)
    z = np.linspace(-1.5,1.5,grid_res)

    X,Y,Z = np.meshgrid(x,y,z,indexing="ij")

    xyz_np = np.stack([X,Y,Z],axis=-1)
    xyz = torch.tensor(xyz_np,dtype=torch.float32).reshape(-1,3).to(device)

    print("[INFO] Training model...")
    model.train(grid=xyz)
    print("[INFO] Training complete.")


    sigma = (3.0) / 3.0 # domain-scale rule

    # Latents
    latent_list=[]
    for i in range(num_scenes):
        key=f"{model_name.lower()}_sphere_{i}"
        latent_list.append(model.trained_scenes[key].get_latent_vector())

    latents=torch.stack(latent_list).float()

    positions_np=np.array(positions)
    latent_values=latents.cpu().numpy().flatten()


    # Visualization grid
    grid_res=128

    x=np.linspace(-1,2,grid_res)
    y=np.linspace(-1.5,1.5,grid_res)
    z=np.linspace(-1.5,1.5,grid_res)

    X,Y,Z=np.meshgrid(x,y,z,indexing="ij")

    xyz_np=np.stack([X,Y,Z],axis=-1)
    xyz=torch.from_numpy(xyz_np).float().reshape(-1,3).to(device)



    # Colormap
    cdict={
        "red":[(0,0,0),(0.5,1,1),(1,1,1)],
        "green":[(0,0,0),(0.5,1,1),(1,0,0)],
        "blue":[(0,1,1),(0.5,1,1),(1,0,0)]
    }

    cmap=mcolors.LinearSegmentedColormap("sdf",cdict)
    norm=TwoSlopeNorm(vmin=-0.6,vcenter=0,vmax=0.6)

    # SDF evaluation
    def eval_sdf(latent_vec):
        out=model.compute_sdf_from_latent(
            latent_vector=latent_vec,
            xyz=xyz
        )
        return out.detach().cpu().numpy().reshape(grid_res,grid_res,grid_res)
    
    def eval_true_sdf(cx,cy):
        out=analytic_sphere_sdf(
            cx=cx,
            cy=cy,
            r=R,
            xyz=xyz
        )
        return out.detach().cpu().numpy().reshape(grid_res,grid_res,grid_res)



    # GIF buffers
    frames_sdf=[]
    frames_error=[]
    frames_sdf3d=[]
    frames_error3d=[]

    # Interpolation
    interp_steps=50
    t_vals=np.linspace(0,1,interp_steps)

    z0=latents[0].to(device)
    z1=latents[-1].to(device)

    sigma=1.0


    #mean guassian weighted error among trained scenes
    Escenes = []

    def gaussian_weighted_error_torch(f_pred, f_true, sigma, N=100_000):
        # GPU sampling
        x = (torch.rand(N, device=device) * 3.0) - 1.0
        y = (torch.rand(N, device=device) * 3.0) - 1.5
        z = (torch.rand(N, device=device) * 3.0) - 1.5

        pts = torch.stack([x, y, z], dim=1)

        ft = f_true(pts)   # torch
        fp = f_pred(pts)   # torch

        w = torch.exp(-(ft ** 2) / (2 * sigma ** 2))

        return (w * (ft - fp) ** 2).sum() / w.sum()


    for i in range(num_scenes):

        z_scene = latents[i].detach().to(device)

        cx = torch.tensor(positions[i], device=device)
        cy = torch.tensor(positions[i] ** 2, device=device)

        def f_pred(pts):
            with torch.no_grad():
                return model.compute_sdf_from_latent(
                    latent_vector=z_scene,
                    xyz=pts
                ).squeeze()


        Escene = gaussian_weighted_error_torch(
            f_pred,
            lambda pts: analytic_sphere_sdf(cx=cx,cy=cy,r=R,xyz=pts),
            sigma=torch.tensor(sigma, device=device)
        )

        Escenes.append(Escene)

    Escenes = torch.stack(Escenes)

    print(f"[METRIC] mean gaussian weighted error among reconstructed scenes = {Escenes.mean().item():.6f}")

    E_t= []

    for i,t in enumerate(t_vals):
        

        zt=(1-t)*z0 + t*z1
        cx=t
        cy=t**2

        sdf=eval_sdf(zt)
        true_sdf = eval_true_sdf(cx=cx,cy=cy)


        def f_pred_interp(pts):
            with torch.no_grad():
                return model.compute_sdf_from_latent(
                    latent_vector=zt,
                    xyz=pts
                ).squeeze()

      

        gaussian_weighted_err = gaussian_weighted_error_torch(
            f_pred_interp,
            lambda pts: analytic_sphere_sdf(cx=cx,cy=cy,r=R,xyz=pts),
            sigma=torch.tensor(sigma, device=device)
        )

  
        E_t.append(gaussian_weighted_err.item())
        

        error=sdf-true_sdf

        # 2D slice
        sdf_slice=sdf[:,:,grid_res//2]
        err_slice=error[:,:,grid_res//2]

        fig,ax=plt.subplots()
        im=ax.imshow(sdf_slice,cmap=cmap,norm=norm,origin="lower")
        plt.colorbar(im)
        path=os.path.join(FRAME_DIR,f"sdf_{i:03d}.png")
        plt.savefig(path)
        plt.close()
        frames_sdf.append(imageio.imread(path))


        fig,ax=plt.subplots()
        im=ax.imshow(err_slice,cmap=cmap,norm=norm,origin="lower")
        plt.colorbar(im)
        path=os.path.join(FRAME_DIR,f"err_{i:03d}.png")
        plt.savefig(path)
        plt.close()
        frames_error.append(imageio.imread(path))


        # 3D point cloud
        step=6

        xs=X[::step,::step,::step].reshape(-1)
        ys=Y[::step,::step,::step].reshape(-1)
        zs=Z[::step,::step,::step].reshape(-1)

        vals=sdf[::step,::step,::step].reshape(-1)
        err_vals=error[::step,::step,::step].reshape(-1)


        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(111,projection='3d')
        sc=ax.scatter(xs,ys,zs,c=vals,cmap=cmap,norm=norm,s=2)
        plt.colorbar(sc)
        path=os.path.join(FRAME_DIR,f"sdf3d_{i:03d}.png")
        plt.savefig(path)
        plt.close()
        frames_sdf3d.append(imageio.imread(path))


        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(111,projection='3d')
        sc=ax.scatter(xs,ys,zs,c=err_vals,cmap=cmap,norm=norm,s=2)
        plt.colorbar(sc)
        path=os.path.join(FRAME_DIR,f"err3d_{i:03d}.png")
        plt.savefig(path)
        plt.close()
        frames_error3d.append(imageio.imread(path))
    

    E_t = np.array(E_t)

    # Integral over t
    import scipy.integrate as integrate
    integral_error = integrate.simpson(E_t, t_vals)
    
    # integral_error = np.trapz(E_t, t_vals)
    print(f"[METRIC] ∫ E(t) dt = {integral_error:.6f}")

    # GIF generation
    imageio.mimsave(os.path.join(EXPERIMENT_ROOT,"sdf_slice.gif"),frames_sdf,duration=0.05)

    imageio.mimsave(os.path.join(EXPERIMENT_ROOT,"error_slice.gif"),frames_error,duration=0.05)

    imageio.mimsave(os.path.join(EXPERIMENT_ROOT,"sdf_3d.gif"),frames_sdf3d,duration=0.05)

    imageio.mimsave(os.path.join(EXPERIMENT_ROOT,"error_3d.gif"),frames_error3d,duration=0.05)


    print("\n[INFO] ---- Metrics ----")
    print("Spearman:",spearmanr(positions_np,latent_values)[0])
    print("Kendall:",kendalltau(positions_np,latent_values)[0])

    reg=LinearRegression()
    reg.fit(latent_values.reshape(-1,1),positions_np)
    print("R^2:",reg.score(latent_values.reshape(-1,1),positions_np))


if __name__=="__main__":
    main()