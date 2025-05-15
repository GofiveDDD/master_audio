#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import joblib


# In[9]:


#1. 数据对齐与保存 Δmesh 数据（嘴部提取）
def align_audio_mesh(data_verts_path, audio_pkl_path, idx_map_path, template_path, output_path):
    import numpy as np, pickle
    from tqdm import tqdm

    data_verts = np.load(data_verts_path)
    with open(audio_pkl_path, 'rb') as f:
        audio_data = pickle.load(f, encoding='latin1')
    with open(idx_map_path, 'rb') as f:
        subj_seq_to_idx = pickle.load(f, encoding='latin1')
    with open(template_path, 'rb') as f:
        template_dict = pickle.load(f, encoding='latin1')

    paired_audio, paired_dmesh, paired_name = [], [], []
    for subject_id in tqdm(subj_seq_to_idx, desc="Aligning data"):
        if subject_id not in audio_data or subject_id not in template_dict:
            continue
        template_mesh = template_dict[subject_id]  # (5023, 3)

        for sent_id in subj_seq_to_idx[subject_id]:
            if sent_id not in audio_data[subject_id]:
                continue
            frame_map = subj_seq_to_idx[subject_id][sent_id]
            sorted_idx = [frame_map[i] for i in sorted(frame_map)]

            expr_seq = data_verts[sorted_idx]  # (T, 5023, 3)
            audio_seq = audio_data[subject_id][sent_id]['audio'].reshape(len(frame_map), -1)  # (T, 464)

            T = min(len(expr_seq), len(audio_seq))
            dmesh_seq = expr_seq[:T] - template_mesh
            paired_audio.append(audio_seq[:T])
            paired_dmesh.append(dmesh_seq)
            paired_name.append(f"{subject_id}_{sent_id}")

    with open(output_path, 'wb') as f:
        pickle.dump({'audio': paired_audio, 'dmesh': paired_dmesh, 'name': paired_name}, f)
        
        
 #2. 提取嘴部索引（只需要做一次）


def extract_mouth_indices(template_ply_path, save_path="mouth_indices.npy", visualize=False):
    import numpy as np, trimesh, matplotlib.pyplot as plt
    mesh = trimesh.load(template_ply_path, process=False)
    v = mesh.vertices

    mouth_idx = [i for i, (_, y, z) in enumerate(v) if -0.06 < y < 0.00 and z > 0.02]
    np.save(save_path, np.array(mouth_idx))
    print(f"✅ Mouth vertices: {len(mouth_idx)}")
    
#3. 裁剪嘴部、保存裁剪后 Δmesh    

def crop_to_mouth(dmesh_path, mouth_idx_path, output_path):
    import numpy as np, pickle
    with open(dmesh_path, 'rb') as f:
        data = pickle.load(f)
    mouth_idx = np.load(mouth_idx_path)

    cropped_dmesh = [seq[:, mouth_idx, :] for seq in data['dmesh']]
    out = {'audio': data['audio'], 'dmesh_mouth': cropped_dmesh, 'name': data['name']}
    with open(output_path, 'wb') as f:
        pickle.dump(out, f)

#4. 执行 PCA 并保存主成分和均值
        
def compute_pca(dmesh_list, save_dir, var_threshold=0.99):
    import numpy as np, os
    from sklearn.decomposition import PCA
    os.makedirs(save_dir, exist_ok=True)

    all_flat = np.concatenate([seq.reshape(seq.shape[0], -1) for seq in dmesh_list], axis=0)
    pca_full = PCA().fit(all_flat)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_dim = np.searchsorted(cum_var, var_threshold) + 1

    pca = PCA(n_components=n_dim).fit(all_flat)
    np.save(os.path.join(save_dir, 'pca_components.npy'), pca.components_)
    np.save(os.path.join(save_dir, 'pca_mean.npy'), pca.mean_)
    return pca
#5. 编码 Δmesh → PCA

def encode_to_pca(dmesh_list, pca):
    code_list = []
    for seq in dmesh_list:
        flat = seq.reshape(seq.shape[0], -1)
        code = (flat - pca.mean_) @ pca.components_.T
        code_list.append(code.astype(np.float32))
    return code_list

#6. 标准化编码并保存 Scaler

def standardize_pca_codes(code_list, save_path):
    from sklearn.preprocessing import StandardScaler
    import numpy as np, joblib, os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_codes = np.vstack(code_list)
    scaler = StandardScaler().fit(all_codes)
    joblib.dump(scaler, save_path)
    scaled = [scaler.transform(seq) for seq in code_list]
    return scaled

#7. 构建 Dataset 和 DataLoader

class SlidingWindowPcaDataset(torch.utils.data.Dataset):
    def __init__(self, audios, codes, window_size=30, stride=5):
        self.samples = []
        for a_seq, c_seq in zip(audios, codes):
            T = min(len(a_seq), len(c_seq))
            for start in range(0, T - window_size + 1, stride):
                self.samples.append((
                    torch.tensor(a_seq[start:start+window_size], dtype=torch.float32),
                    torch.tensor(c_seq[start:start+window_size], dtype=torch.float32)
                ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def build_loaders(audio_list, code_list, seed=42, win=30, stride=5, batch=16):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    a_train, a_temp, c_train, c_temp = train_test_split(audio_list, code_list, test_size=0.1, random_state=seed)
    a_val, a_test, c_val, c_test = train_test_split(a_temp, c_temp, test_size=0.2, random_state=seed)

    train_ds = SlidingWindowPcaDataset(a_train, c_train, win, stride)
    val_ds   = SlidingWindowPcaDataset(a_val, c_val, win, stride)
    test_ds  = SlidingWindowPcaDataset(a_test, c_test, win, stride)

    return (
        DataLoader(train_ds, batch_size=batch, shuffle=False, drop_last=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False, drop_last=True),
        DataLoader(test_ds, batch_size=batch, shuffle=False, drop_last=True),
    )

