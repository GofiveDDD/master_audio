#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# In[ ]:


class LipsyncTransformer(nn.Module):
    def __init__(self, audio_dim=464, mouth_dim=31, n_heads=4, n_layers=2,
                 ff_dim=112, dropout=0.2, max_seq_len=30):
        super().__init__()
        self.input_proj = nn.Linear(audio_dim, ff_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, ff_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(ff_dim, mouth_dim)

    def forward(self, audio):
        B, T, _ = audio.shape
        x = self.input_proj(audio)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        return self.output_proj(x)


class VAEEncoder(nn.Module):
    def __init__(self, input_dim=31, latent_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        mean, log_var = torch.chunk(x, 2, dim=-1)
        return mean, log_var


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=31):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.fc(z)


class TransformerVAE(nn.Module):
    def __init__(self, transformer_model, vae_encoder, vae_decoder):
        super().__init__()
        self.transformer = transformer_model
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

    def forward(self, audio_input):
        transformer_output = self.transformer(audio_input)
        mean, log_var = self.vae_encoder(transformer_output)
        latent = self.reparameterize(mean, log_var)
        return self.vae_decoder(latent)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


# ---------- train_model----------
def train_model(model, train_loader, val_loader, optimizer, num_epochs=30,
                device='cuda', patience=20, save_path='best_model_vae.pt',
                initial_lr=1e-4):
    model = model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        new_lr = initial_lr * (1 - epoch / num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(new_lr, 1e-6)

        model.train()
        total_train_loss = 0.0
        for audio, expr in train_loader:
            audio, expr = audio.to(device), expr.to(device)
            pred = model(audio)
            loss = ((pred - expr) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for audio, expr in val_loader:
                audio, expr = audio.to(device), expr.to(device)
                pred = model(audio)
                loss = ((pred - expr) ** 2).mean()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # 

        # Check validation loss and update the model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Validation loss decreased, model saved to {save_path} (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ Validation loss did not improve (val_loss={avg_val_loss:.4f}), patience counter: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️ Validation loss did not improve for {patience} consecutive epochs, stopping training early.")
            break

    return train_losses, val_losses


# ---------- plot ----------
def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(4, 2))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# ---------- evaluation ----------
def evaluate_on_testset(model, test_loader, device='cuda', save_dir='output_expr_vae_eval'):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_cos_sim = 0.0
    total_smoothness = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for audio, expr in tqdm(test_loader, desc="Evaluating"):
            audio, expr = audio.to(device), expr.to(device)
            pred = model(audio)

            mse = F.mse_loss(pred, expr).item()
            mae = F.l1_loss(pred, expr).item()
            cos_sim = F.cosine_similarity(
                F.normalize(pred, dim=-1),
                F.normalize(expr, dim=-1),
                dim=-1
            ).mean().item()
            smoothness = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean().item()

            total_mse += mse
            total_mae += mae
            total_cos_sim += cos_sim
            total_smoothness += smoothness
            n_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(expr.cpu())

    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches
    avg_cos_sim = total_cos_sim / n_batches
    avg_smooth = total_smoothness / n_batches

    print("\n Assessment results:")
    print(f"✅ MSE:              {avg_mse:.4f}")
    print(f"✅ MAE:              {avg_mae:.4f}")
    print(f"✅ Cosine Similarity:{avg_cos_sim:.4f}")
    print(f"✅ Smoothness:       {avg_smooth:.4f}")

    pred_all = torch.cat(all_preds).numpy()
    target_all = torch.cat(all_targets).numpy()
    np.save(f"{save_dir}/predicted_expr.npy", pred_all)
    np.save(f"{save_dir}/groundtruth_expr.npy", target_all)

    with open(f"{save_dir}/metrics.txt", "w") as f:
        f.write(f"MSE: {avg_mse:.6f}\n")
        f.write(f"MAE: {avg_mae:.6f}\n")
        f.write(f"CosineSimilarity: {avg_cos_sim:.6f}\n")
        f.write(f"Smoothness: {avg_smooth:.6f}\n")

    return {
        "MSE": avg_mse,
        "MAE": avg_mae,
        "CosineSimilarity": avg_cos_sim,
        "Smoothness": avg_smooth,
    }

