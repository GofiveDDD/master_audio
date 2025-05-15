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


# ---------- 模型定义 ----------
class LipsyncTransformer(nn.Module):
    def __init__(self, 
                 audio_dim=464,
                 mouth_dim=31,
                 n_heads=4, # n_heads要能整除 d_model（ff_dim）
                 n_layers=2,# 一般选 2–4 层 做 baseline
                 ff_dim=112, # ff_dim = mouth_dim*2或者4
                 dropout=0.2, # 过拟合严重可提高到 0.3
                 max_seq_len=30):  # 支持最大序列长度
        super().__init__()

        # 1. 投影输入特征
        self.input_proj = nn.Linear(audio_dim, ff_dim)  # [B, T, 464] -> [B, T, ff_dim]

        # 2. 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, ff_dim))  # [1, T, ff_dim]

        # 3. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim, 
            nhead=n_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. 输出层
        self.output_proj = nn.Linear(ff_dim, mouth_dim)  # [B, T, ff_dim] -> [B, T, 28]

    def forward(self, audio):
        """
        audio: [B, T, 464]
        """
        B, T, _ = audio.shape

        # 1. 输入特征投影
        x = self.input_proj(audio)  # [B, T, ff_dim]

        # 2. 加位置编码（注意裁剪 pos_embed 到当前 T）
        x = x + self.pos_embed[:, :T, :]  # [B, T, ff_dim]

        # 3. Transformer 编码
        x = self.transformer(x)  # [B, T, ff_dim]

        # 4. 映射到输出
        out = self.output_proj(x)  # [B, T, 28]

        return out



# ---------- 训练函数 ----------
def train_model(model, train_loader, val_loader, optimizer,
                num_epochs=30, device='cuda', patience=10, save_path='best_transformer.pt',
                initial_lr=1e-4):
    
    model = model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    for epoch in range(num_epochs):
        # -------- 手动线性下降学习率 --------
        new_lr = initial_lr * (1 - epoch / num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(new_lr, 1e-6)  # 防止变成0

        # -------- 训练 --------
        model.train()
        total_train_loss = 0.0
        for audio, expr in train_loader:
            audio = audio.to(device)
            expr = expr.to(device)

            pred = model(audio)
            loss = ((pred - expr) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- 验证 --------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for audio, expr in val_loader:
                audio = audio.to(device)
                expr = expr.to(device)

                pred = model(audio)
                loss = ((pred - expr) ** 2).mean()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # -------- 打印信息 --------

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

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



# ---------- 绘图 ----------
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


# ---------- 评估函数 ----------
def evaluate_on_testset(model, test_loader, device='cuda', save_dir='output_expr_trans_eval'):
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

    print("\n📊 Assessment results:")
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

