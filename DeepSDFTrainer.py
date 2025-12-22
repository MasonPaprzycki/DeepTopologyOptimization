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


# -----------------------------
# DeepSDF Network
# -----------------------------
class DeepSDF(nn.Module):
    """DeepSDF MLP with latent injection at input and mid-network."""
    def __init__(self, input_dim, latent_dim=256, hidden_dim=512,
                 num_layers=8, latent_injection_layer=4, soft_latent=True):
        super().__init__()
        self.latent_injection_layer = latent_injection_layer
        self.soft_latent = soft_latent

        layers = nn.ModuleList()
        layers.append(nn.Linear(input_dim + latent_dim, hidden_dim))
        for i in range(1, num_layers):
            if i == latent_injection_layer:
                layers.append(nn.Linear(hidden_dim + latent_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = layers
        self.final = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus(beta=100) if soft_latent else nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, z):
        h = torch.cat([x, z], dim=1)
        for i, layer in enumerate(self.layers):
            if i == self.latent_injection_layer:
                h = torch.cat([h, z], dim=1)
            h = self.activation(layer(h))
        return self.final(h)


# -----------------------------
# Dataset
# -----------------------------
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


# -----------------------------
# Trainer
# -----------------------------
class DeepSDFTrainer:
    """Trainer for DeepSDF auto-decoder."""
    def __init__(self, base_directory, model, num_shapes, latent_dim=256, sigma0=1e-4,
                 lr_net=5e-4, lr_latent=1e-3, clamp_delta=None, device="cpu",regularize_latent: bool = True):
        
        self.base_directory = base_directory
        self.device = device
        self.regularize_latent = regularize_latent
        self.model = model.to(device)
        self.sigma0 = sigma0
        self.clamp_delta = clamp_delta
        self.save_dir = os.path.join(base_directory, "snapshots")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize latent codes
        self.latents = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.latents.weight, mean=0.0, std=0.01)
        self.latents = self.latents.to(device)

        # Optimizer: separate learning rates for network and latents
        self.optimizer = optim.Adam([
            {"params": self.model.parameters(), "lr": lr_net},
            {"params": self.latents.parameters(), "lr": lr_latent},
        ])

        # Loss history for plotting
        self.loss_history = {"total": [], "data": [], "latent_reg": []}

    def train_step(self, shape_ids, points, sdf, sigma):
        """Single training step implementing Eq. (6)."""
        B, N, D = points.shape
        shape_ids = shape_ids.to(self.device)
        points = points.to(self.device)
        sdf = sdf.to(self.device)

        # --- Latent per shape ---
        z_shape = self.latents(shape_ids)  # (B, latent_dim)

        # --- Latent regularization ---
        latent_reg = sigma * (z_shape ** 2).sum() if self.regularize_latent else torch.tensor(0.0, device=self.device)

        # --- Expand latent to per-point ---
        z = z_shape[:, None, :].expand(B, N, -1).reshape(-1, z_shape.shape[1])
        x = points.reshape(-1, D)
        s = sdf.reshape(-1, 1)

        # --- Data term ---
        pred = self.model(x, z)
        data_loss = clamped_l1_loss(pred, s, self.clamp_delta).sum() if self.clamp_delta is not None else l1_loss(pred, s).sum()

        # --- Total loss ---
        loss = data_loss + latent_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), data_loss.item(), latent_reg.item()

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

    def train(self, dataloader, epochs, snapshot_every=100):
        """Full training loop with logging and loss tracking."""
        for epoch in range(1, epochs + 1):
            sigma = self.sigma0 * min(1.0, 1.0 / epoch)
            epoch_total, epoch_data, epoch_latent = 0.0, 0.0, 0.0

            for sid, pts, sdf in dataloader:
                loss, data_loss, latent_reg = self.train_step(sid, pts, sdf, sigma)
                epoch_total += loss
                epoch_data += data_loss
                epoch_latent += latent_reg

            # Store averaged losses for plotting
            self.loss_history["total"].append(epoch_total / len(dataloader))
            self.loss_history["data"].append(epoch_data / len(dataloader))
            self.loss_history["latent_reg"].append(epoch_latent / len(dataloader))

            if epoch % snapshot_every == 0:
                self.save_snapshot(epoch)
                print(f"[{epoch:04d}] total_loss={epoch_total:.6e} "
                    f"data_loss={epoch_data:.6e} latent_reg={epoch_latent:.6e}")
                
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


# -----------------------------
# Latent inference
# -----------------------------
def infer_latent(model: DeepSDF, points: torch.Tensor, sdf: torch.Tensor,
                 latent_dim: int = 256, latent_sigma: float = 0.01,
                 lr: float = 1e-3, iters: int = 800, clamp_delta: float = 0.1,
                 device: str = "cpu"):
    """Optimize a latent vector for a new shape with a fixed network."""
    model.eval()
    points = points.to(device)
    sdf = sdf.to(device)

    z = torch.zeros((1, latent_dim), device=device, requires_grad=True)
    nn.init.normal_(z, mean=0.0, std=latent_sigma)

    optimizer = optim.Adam([z], lr=lr)

    for _ in range(iters):
        z_rep = z.expand(points.shape[0], -1)
        pred = model(points, z_rep)

        data_loss = clamped_l1_loss(pred, sdf, clamp_delta).mean() if clamp_delta is not None else l1_loss(pred, sdf).mean()
        latent_reg = (z ** 2).sum() / (latent_sigma ** 2)
        loss = data_loss + latent_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return z.detach()
