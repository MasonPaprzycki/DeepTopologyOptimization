import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from typing import Dict, Tuple
import os
import matplotlib.pyplot as plt

#utilities 

def clamp_sdf(x, delta=0.1):
    return torch.clamp(x, -delta, delta)


def clamped_l1_loss(pred, target, delta=0.1):
    return torch.abs(
        clamp_sdf(pred, delta) - clamp_sdf(target, delta)
    )


# -----------------------------
# DeepSDF Network
# -----------------------------

class DeepSDF(nn.Module):
    """
    DeepSDF auto-decoder MLP.

    f_theta(x, z) -> sdf

    - Geometry (xyz + operator params) processed first
    - Scene latent injected at input and again mid-network
    - Latent acts as a basis coefficient, not a bias
    """

    def __init__(
        self,
        input_dim: int,          # xyz + operator params
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        latent_injection_layer: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert latent_injection_layer < num_layers

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.latent_injection_layer = latent_injection_layer

        layers = nn.ModuleList()

        # Layer 0: geometry + latent
        layers.append(
            nn.Linear(input_dim + latent_dim, hidden_dim)
        )

        # Hidden layers
        for i in range(1, num_layers):
            if i == latent_injection_layer:
                layers.append(
                    nn.Linear(hidden_dim + latent_dim, hidden_dim)
                )
            else:
                layers.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )

        self.layers = layers
        self.final = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(
                    m.weight,
                    mean=0.0,
                    std=math.sqrt(2) / math.sqrt(m.out_features),
                )
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        x: (B, input_dim)
        z: (B, latent_dim)
        """

        # First latent injection
        h = torch.cat([x, z], dim=1)

        for i, layer in enumerate(self.layers):
            if i == self.latent_injection_layer:
                # Second latent injection
                h = torch.cat([h, z], dim=1)

            h = layer(h)
            h = self.activation(h)
            h = self.dropout(h)

        return self.final(h)

class SDFDataset(Dataset):
    """
    data[shape_id] = (points[N, D], sdf[N, 1])
    """

    def __init__(self, data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        self.data = data
        self.shape_ids = list(data.keys())

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, idx):
        sid = self.shape_ids[idx]
        pts, sdf = self.data[sid]
        return sid, pts, sdf


class DeepSDFTrainer:
    def __init__(
        self,
        base_directory: str,
        model: DeepSDF,
        num_shapes: int,
        latent_dim: int = 256,
        latent_sigma: float = 0.01,
        lr_network: float = 1e-4,
        lr_latent: float = 1e-3,
        clamp_delta: float = 0.1,
        device: str = "cpu"
    ):
        self.device = device
        self.model = model.to(device)
        self.latent_sigma = latent_sigma
        self.clamp_delta = clamp_delta

        # Latent codes
        self.latents = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.latents.weight, mean=0.0, std=latent_sigma)
        self.latents = self.latents.to(device)

        self.optimizer = optim.Adam(
            [
                {"params": self.model.parameters(), "lr": lr_network},
                {"params": self.latents.parameters(), "lr": lr_latent},
            ]
        )

        # Create save directory
        self.save_dir = os.path.join(base_directory, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # For logging
        self.loss_history = {"total": [], "data": [], "latent_reg": []}

    def train_step(self, shape_ids, points, sdf_gt):
        B, N, D = points.shape

        shape_ids = shape_ids.to(self.device)
        points = points.to(self.device).view(-1, D)
        sdf_gt = sdf_gt.to(self.device).view(-1, 1)

        # Latent expansion per-point
        z = self.latents(shape_ids)
        z = z.unsqueeze(1).expand(B, N, -1).reshape(-1, z.shape[-1])

        sdf_pred = self.model(points, z)
        data_loss = clamped_l1_loss(sdf_pred, sdf_gt, self.clamp_delta).mean()
        latent_reg = (z ** 2).sum(dim=1).mean() / (self.latent_sigma ** 2)
        loss = data_loss + latent_reg

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.latents.parameters(), 1.0)
        self.optimizer.step()

        return {
            "total": loss.item(),
            "data": data_loss.item(),
            "latent_reg": latent_reg.item(),
        }

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

    def plot_losses(self):
        """Plot loss curves over training epochs."""
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

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 500,
        snapshot_every: int = 1,
    ):
        """Full training loop with snapshot saving and logging."""
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for shape_ids, points, sdf in dataloader:
                stats = self.train_step(shape_ids, points, sdf)
                epoch_loss += stats["total"]

            epoch_loss /= len(dataloader)
            self.loss_history["total"].append(epoch_loss)
            self.loss_history["data"].append(stats["data"])
            self.loss_history["latent_reg"].append(stats["latent_reg"])

            if epoch % snapshot_every == 0:
                print(
                    f"[EPOCH {epoch:04d}] "
                    f"loss={epoch_loss:.6f} "
                    f"data={stats['data']:.6f} "
                    f"latent regression={stats['latent_reg']:.6f}"
                )
                self.save_snapshot(epoch )
            
        # Save final loss curve
        self.plot_losses()


def infer_latent(
    model: DeepSDF,
    points: torch.Tensor,
    sdf: torch.Tensor,
    latent_dim: int = 256,
    latent_sigma: float = 0.01,
    lr: float = 1e-3,
    iters: int = 800,
    clamp_delta: float = 0.1,
    device: str = "cpu",
):
    """
    Solves:
        argmin_z sum L(f(z,x), s) + ||z||^2 / sigma^2
    """

    model.eval()
    points = points.to(device)
    sdf = sdf.to(device)

    z = torch.zeros((1, latent_dim), device=device, requires_grad=True)
    nn.init.normal_(z, mean=0.0, std=latent_sigma)

    optimizer = optim.Adam([z], lr=lr)

    for _ in range(iters):
        z_rep = z.expand(points.shape[0], -1)
        pred = model(points, z_rep)

        data_loss = clamped_l1_loss(
            pred, sdf, clamp_delta
        ).mean()

        latent_reg = (z ** 2).sum() / (latent_sigma ** 2)

        loss = data_loss + latent_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return z.detach()


