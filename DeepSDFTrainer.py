import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from typing import Dict, Tuple
import os
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def clamp_sdf(x, delta=0.1):
    """Clamp SDF to [-delta, delta] to focus on near-surface points."""
    return torch.clamp(x, -delta, delta)


def clamped_l1_loss(pred, target, delta=0.1):
    """Clamped L1 loss, summing differences after clamping."""
    return torch.abs(clamp_sdf(pred, delta) - clamp_sdf(target, delta))

def l1_loss(pred, target):
    """Standard L1 loss."""
    return torch.abs(pred - target)


class DeepSDF(nn.Module):
    """DeepSDF MLP with latent injection at input and optionally at mid-network."""
    def __init__(self, input_dim, latent_dim=256, hidden_dim=512,
                num_layers=8, skip_layer=None, soft_latent=False,weight_norm=False):
        super().__init__()

        self.skip_layer = skip_layer
        self.soft_latent = soft_latent
        self.weight_norm = weight_norm

        self.layers = nn.ModuleList()

        # First layer: ALWAYS inject latent
        self.layers.append(nn.utils.weight_norm(nn.Linear(input_dim + latent_dim, hidden_dim))) if weight_norm else self.layers.append(nn.Linear(input_dim + latent_dim, hidden_dim))

        # Hidden layers
        for i in range(1, num_layers):
            if skip_layer is not None and i == skip_layer:
                self.layers.append(nn.utils.weight_norm(nn.Linear(hidden_dim+ input_dim + latent_dim, hidden_dim))) if self.weight_norm else self.layers.append(nn.Linear(hidden_dim+ input_dim + latent_dim, hidden_dim))
            else:
                self.layers.append(nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim))) if self.weight_norm else self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.final = nn.utils.weight_norm(nn.Linear(hidden_dim, 1)) if self.weight_norm else nn.Linear(hidden_dim, 1)

        self.activation = nn.Softplus(beta=100) if soft_latent else nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, z):
        """
        Forward pass for DeepSDF.

        Args:
            x: (B, input_dim) coordinate points
            z: (B, latent_dim) latent vector
        Returns:
            (B, 1) SDF predictions
        """
        # Always inject latent at input
        h = torch.cat([x, z], dim=1)

        for i, layer in enumerate(self.layers):
            # Inject latent at hidden layer only if specified
            if self.skip_layer is not None and i == self.skip_layer:
                h = torch.cat([h,x, z], dim=1)
            h = self.activation(layer(h))

        return self.final(h)


class SDFDataset(Dataset):
    """Dataset for shapes: each item is (shape_id, points[N,D], sdf[N,1])"""
    def __init__(self, data):
        self.shape_ids = list(data.keys())
        self.data = data

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, idx):
        sid = self.shape_ids[idx]
        pts, sdf = self.data[sid]
        return sid, pts, sdf



class DeepSDFTrainer:
    """Trainer for DeepSDF auto-decoder."""
    def __init__(self, base_directory, model, num_shapes, latent_dim=256, sigma0=1e-4,
                 lr_net=5e-4, lr_latent=1e-3, clamp_delta:float|None=0.1,regularize_latent: bool = True, weight_norm: bool = False, subsample_clamp_dist=0.1, device=None):
        
        self.base_directory = base_directory
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.regularize_latent = regularize_latent
        self.model = model.to(self.device)
        self.sigma0 = sigma0
        self.clamp_delta = clamp_delta
        self.save_dir = os.path.join(base_directory, "snapshots")
        self.subsample_clamp_dist = subsample_clamp_dist
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize latent codes
        self.latents = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.latents.weight, mean=0.0, std=0.01)
        self.latents = self.latents.to(self.device)

        # Optimizer: separate learning rates for network and latents
        self.optimizer = optim.Adam([
            {"params": self.model.parameters(), "lr": lr_net},
            {"params": self.latents.parameters(), "lr": lr_latent},
        ])

        # Loss history for plotting
        self.loss_history = {"total": [], "data": [], "latent_reg": []}

    def train_step(
            self, 
            shape_ids,
            points_pos,
            points_neg,
            points_pos_outlier,
            points_neg_outlier,
            sdf_pos, 
            sdf_neg, 
            sdf_pos_outlier, 
            sdf_neg_outlier, 
            sigma):
        """
        Single training step for a batch of shapes.

        Args:
            shape_ids: (B,) tensor of shape indices
            points: (B, N, D) tensor of input coordinates
            sdf: (B, N, 1) tensor of SDF values
            sigma: float, latent regularization weight

        Returns:
            loss, data_loss, latent_reg
        """
        # Move to device
        points_pos = points_pos.to(self.device)
        points_neg = points_neg.to(self.device)
        points_pos_outlier = points_pos_outlier.to(self.device)
        points_neg_outlier = points_neg_outlier.to(self.device)

        #points = points.to(self.device)
   
        sdf_pos= sdf_pos.to(self.device)
        sdf_neg= sdf_neg.to(self.device)
        sdf_pos_outlier =sdf_pos_outlier.to(self.device)
        sdf_neg_outlier = sdf_neg_outlier.to(self.device)

        shape_ids = shape_ids.to(self.device)
      
        B_pos, N_pos, D_pos = points_pos.shape
        B_neg, N_neg, D_neg = points_neg.shape
        B_pos_out, N_pos_out, D_pos_out = points_pos_outlier.shape
        B_neg_out, N_neg_out, D_neg_out = points_neg_outlier.shape

        # SAFE lengths for indexing
        N_pos = sdf_pos.shape[1]
        N_neg = sdf_neg.shape[1]
        N_pos_out = sdf_pos_outlier.shape[1]
        N_neg_out = sdf_neg_outlier.shape[1]

        total_target_samples = 5000
        near_ratio = 0.95
        far_ratio = 1.0 - near_ratio
        half_samples =int(total_target_samples * near_ratio / 2)
        pos_num_samples = min(half_samples, N_pos)
        neg_num_samples = min(half_samples, N_neg)
        far_half_samples = int(total_target_samples * far_ratio / 2)
        pos_out_num_samples = min(far_half_samples, N_pos_out)
        neg_out_num_samples = min(far_half_samples, N_neg_out)

        #Random subsampling keeping optimal deepSDF training distribution intact
        sample_idx_pos = torch.randint(0, N_pos, (B_pos, pos_num_samples), device=self.device)
        sample_idx_neg = torch.randint(0, N_neg, (B_neg, neg_num_samples), device=self.device)
        if pos_out_num_samples>0:
            sample_idx_pos_out = torch.randint(0, N_pos_out, (B_pos_out, pos_out_num_samples), device=self.device)
        else:
            sample_idx_pos_out=None

        if neg_out_num_samples > 0: 
            sample_idx_neg_out = torch.randint(0, N_neg_out, (B_neg_out, neg_out_num_samples), device=self.device)
        else:
            sample_idx_neg_out=None

        points_pos = torch.gather(points_pos, 1, sample_idx_pos.unsqueeze(-1).expand(-1, -1, D_pos))
        points_neg = torch.gather(points_neg, 1, sample_idx_neg.unsqueeze(-1).expand(-1, -1, D_neg))
        points_pos_outlier = torch.gather(points_pos_outlier, 1, sample_idx_pos_out.unsqueeze(-1).expand(-1, -1, D_pos_out)) if sample_idx_pos_out is not None else None
        points_neg_outlier = torch.gather(points_neg_outlier, 1, sample_idx_neg_out.unsqueeze(-1).expand(-1, -1, D_neg_out)) if sample_idx_neg_out is not None else None

        sdf_pos = torch.gather(sdf_pos, 1, sample_idx_pos.unsqueeze(-1))
        sdf_neg = torch.gather(sdf_neg, 1, sample_idx_neg.unsqueeze(-1))
        sdf_pos_outlier = torch.gather(sdf_pos_outlier, 1, sample_idx_pos_out.unsqueeze(-1)) if sample_idx_pos_out is not None else None
        sdf_neg_outlier = torch.gather(sdf_neg_outlier, 1, sample_idx_neg_out.unsqueeze(-1)) if sample_idx_neg_out is not None else None
        points = torch.cat([points_pos, points_neg],dim=1)
        sdf = torch.cat([sdf_pos,sdf_neg],dim=1)
        if points_pos_outlier is not None:
            points= torch.cat([points, points_pos_outlier],dim=1)
        
        if points_neg_outlier is not None:
            points = torch.cat([points,points_neg_outlier],dim=1)

        if sdf_pos_outlier is not None:
            sdf = torch.cat([sdf, sdf_pos_outlier], dim=1)
        if sdf_neg_outlier is not None: 
            sdf = torch.cat([sdf, sdf_neg_outlier], dim=1)

        B, N_actual, D = points.shape

        z_shape = self.latents(shape_ids)  # (B, latent_dim)
        z_expanded = z_shape.repeat_interleave(N_actual, dim=0)
        x_flat = points.reshape(B * N_actual, D)
        s_flat = sdf.reshape(B * N_actual, 1)

        # Latent regularization
        latent_reg = sigma * (z_shape ** 2).sum() if self.regularize_latent else torch.tensor(0.0, device=self.device)
        #forward pass 
        pred = self.model(x_flat, z_expanded)
        #compute loss
        if self.clamp_delta is not None:
            data_loss = clamped_l1_loss(pred, s_flat, self.clamp_delta).sum()
        else:
            data_loss = l1_loss(pred, s_flat).sum()

        total_loss = data_loss + latent_reg


        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), data_loss.item(), latent_reg.item()

    def train_step_stochastic_distribution(self, shape_ids, points, sdf, sigma):
    
        "Fast subsampling strategy that breaks training with clamped loss" 

        # ----------------------
        # Random subsampling
        # ----------------------
        points = points.to(self.device)
        B, N, D = points.shape[1]

        num_samples = min(5000, N)
        sample_idx = torch.randint(0, N, (B, num_samples), device=self.device)

        points = torch.gather(points, 1, sample_idx.unsqueeze(-1).expand(-1, -1, D))
        sdf = torch.gather(sdf, 1, sample_idx.unsqueeze(-1))

        # Updated shape after subsampling
        B, N_actual, D = points.shape

        # ----------------------
        # Latent per shape
        # ----------------------
        z_shape = self.latents(shape_ids)  # (B, latent_dim)
        z_expanded = z_shape.repeat_interleave(N_actual, dim=0)

        # ----------------------
        # Flatten points and sdf for MLP
        # ----------------------
        x_flat = points.reshape(B * N_actual, D)
        s_flat = sdf.reshape(B * N_actual, 1)

        # ----------------------
        # Latent regularization
        # ----------------------
        latent_reg = sigma * (z_shape ** 2).sum() if self.regularize_latent else torch.tensor(0.0, device=self.device)

        # ----------------------
        # Forward pass
        # ----------------------
        pred = self.model(x_flat, z_expanded)

        # ----------------------
        # Compute loss
        # ----------------------
        if self.clamp_delta is not None:
            data_loss = clamped_l1_loss(pred, s_flat, self.clamp_delta).sum()
        else:
            data_loss = l1_loss(pred, s_flat).sum()

        total_loss = data_loss + latent_reg

        # ----------------------
        # Backprop
        # ----------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), data_loss.item(), latent_reg.item()
        

    def save_snapshot(self, epoch: int):
        """Save model, latents, and optimizer states for a given epoch."""
        snapshot = {
            "model_state_dict": self.model.state_dict(),
            "latent_codes": self.latents.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        path = os.path.join(self.save_dir, f"snapshot_epoch_{epoch:04d}.pth")
        torch.save(snapshot, path)
        print(f"[INFO] Saved snapshot → {path}")

    def train(self, dataloader, epochs, snapshot_every=100, stochastic_distribution=False):
        """Full training loop with logging and loss tracking."""

        #pull out outliers first its an outlier if abs(sdf) >self.clamp delta
    

        preprocessed = []

        for sid, pts, sdf in dataloader:

            pts = pts.to(self.device)
            sdf = sdf.to(self.device)
            sid = sid.to(self.device)

            if stochastic_distribution==True:
                for epoch in range (1, epochs +1):
                    sigma = self.sigma0 * min(1.0, 1.0 / epoch)
                    epoch_total, epoch_data, epoch_latent = 0.0, 0.0, 0.0
                    for sid, pts, sdf in dataloader: 
                        loss,data_loss,latent_reg = self.train_step_stochastic_distribution(
                            shape_ids=sid, points=pts, sdf= sdf, sigma=sigma
                        )

                    epoch_total += loss
                    epoch_data += data_loss
                    epoch_latent += latent_reg

                    self.loss_history["total"].append(epoch_total / len(preprocessed))
                    self.loss_history["data"].append(epoch_data / len(preprocessed))
                    self.loss_history["latent_reg"].append(epoch_latent / len(preprocessed))
                    print(f"[{epoch:04d}] total_loss={epoch_total:.6e} "
                        f"data_loss={epoch_data:.6e} latent_reg={epoch_latent:.6e}")

                    if epoch % snapshot_every == 0:
                        self.save_snapshot(epoch)
                
                
                self.plot_losses()

                return
            

            B, N, D = pts.shape

            for b in range(B):

                pts_b = pts[b]          # (N, D)
                sdf_b = sdf[b]          # (N, 1)
                sid_b = sid[b:b+1]      # keep batch dim -> (1,)

                sdf_flat = sdf_b[:, 0]  # (N,)

                pos_mask = sdf_flat > 0
                neg_mask = sdf_flat < 0

                if self.clamp_delta is not None:
                    near_mask = torch.abs(sdf_flat) <= self.clamp_delta
                    far_mask  = torch.abs(sdf_flat) > self.clamp_delta
                else:
                    near_mask = torch.ones_like(sdf_flat, dtype=torch.bool)
                    far_mask  = torch.zeros_like(sdf_flat, dtype=torch.bool)

                pos_near = pos_mask & near_mask
                neg_near = neg_mask & near_mask
                pos_far  = pos_mask & far_mask
                neg_far  = neg_mask & far_mask

                # safe slicing (handles empty tensors correctly)
                pts_pos_near = pts_b[pos_near]
                pts_neg_near = pts_b[neg_near]
                pts_pos_far  = pts_b[pos_far]
                pts_neg_far  = pts_b[neg_far]

                sdf_pos_near = sdf_b[pos_near]
                sdf_neg_near = sdf_b[neg_near]
                sdf_pos_far  = sdf_b[pos_far]
                sdf_neg_far  = sdf_b[neg_far]

                # restore batch dimension
                preprocessed.append(
                    (
                        sid_b,
                        pts_pos_near.unsqueeze(0),
                        pts_neg_near.unsqueeze(0),
                        pts_pos_far.unsqueeze(0),
                        pts_neg_far.unsqueeze(0),
                        sdf_pos_near.unsqueeze(0),
                        sdf_neg_near.unsqueeze(0),
                        sdf_pos_far.unsqueeze(0),
                        sdf_neg_far.unsqueeze(0),
                    )
                )
        
        for epoch in range(1, epochs + 1):
            sigma = self.sigma0 * min(1.0, 1.0 / epoch)
            epoch_total, epoch_data, epoch_latent = 0.0, 0.0, 0.0

            for sid, pos_pts, neg_pts, pos_out_pts, neg_out_pts, sdf_pos, sdf_neg, sdf_pos_out, sdf_neg_out in preprocessed:
                loss, data_loss, latent_reg = self.train_step(
                    sid,
                    points_pos=pos_pts,
                    points_neg=neg_pts,
                    points_pos_outlier=pos_out_pts,
                    points_neg_outlier=neg_out_pts,
                    sdf_pos=sdf_pos,
                    sdf_neg=sdf_neg,
                    sdf_pos_outlier=sdf_pos_out,
                    sdf_neg_outlier=sdf_neg_out,
                    sigma=sigma)
                
                epoch_total += loss
                epoch_data += data_loss
                epoch_latent += latent_reg

            self.loss_history["total"].append(epoch_total / len(preprocessed))
            self.loss_history["data"].append(epoch_data / len(preprocessed))
            self.loss_history["latent_reg"].append(epoch_latent / len(preprocessed))
            print(f"[{epoch:04d}] total_loss={epoch_total:.6e} "
                f"data_loss={epoch_data:.6e} latent_reg={epoch_latent:.6e}")

            if epoch % snapshot_every == 0:
                self.save_snapshot(epoch)
               
                
        self.plot_losses()

    def plot_losses(self):
        """Plot the training curves of total, data, and latent losses."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history["total"], label="Total Loss")
        plt.plot(self.loss_history["data"], label="Data Loss")
        plt.plot(self.loss_history["latent_reg"], label="Latent Reg")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("DeepSDF Training Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[INFO] Loss curve saved → {save_path}")