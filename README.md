# Audio-driven 3D facial animation generation — VocaGAN

## Introduction

Based on the VOCA dataset, this project implements the training and evaluation of multiple voice-driven 3D facial animation generation models, including RNN, Transformer, Transformer + VAE, and Transformer + GAN.
The core contribution is the Transformer + GAN framework, which significantly improves the naturalness of animation and voice synchronization accuracy.

Each main program file is an independent training process entry, covering data alignment, feature extraction, model training and testing.

## File description

data_utils.py: Contains data preprocessing functions, such as grid alignment, PCA dimensionality reduction, data normalization, mouth cropping, etc.

RNN_train.py, Transformer_train.py, Transformer_GAN_train.py, Transformer_VAE_train.py: model definition and training modules.

main_transformer_gan.py is the main program of the complete process, including data processing, model training and testing. After running, it starts training the Transformer+GAN model.

## Instructions for use

Request the VOCASET data from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/).

Place the downloaded the Training Data，Files `data_verts.npy`, `processed_audio_deepspeech.pkl`, `templates.pkl`, `subj_seq_to_idx.pkl`will be used for training.

You will also need to download the registration data. Files `FaceTalk_170725_00137_TA/sentence01/sentence01.000001.ply` (Can be replaced by registration dataa or other sentence files). This file is used to extract the vertex index of the mouth from the template mesh file.


```python
python main_transformer_gan.py --data_verts_path D:/master_new/trainningdata/data_verts.npy \
--audio_pkl_path D:/master_new/trainningdata/processed_audio_deepspeech.pkl \
--idx_map_path D:/master_new/trainningdata/subj_seq_to_idx.pkl \
--template_path D:/master_new/trainningdata/templates.pkl \
--template_ply_path D:/voice/registereddata/FaceTalk_170725_00137_TA/sentence01/sentence01.000001.ply \#该文件可替换
--epochs 100 --batch_size 32 --output_model best_model_gan.pt
```

The script will complete the following steps:

1. Align and save the mesh with the audio

2. Extract vertex indices (only need to run once, and save the results)

3. Crop Δmesh data

4. PCA dimension reduction and data normalization

5. Build data loader

6. Model training and validation

7. Plot training curves

8. Test set evaluation

3. Result visualization

## Parameter description


```python
--data_verts_path: raw vertex data path

--audio_pkl_path: audio feature data path

--idx_map_path: data index mapping file path

--template_path: template mesh file path

--template_ply_path: PLY file path for extracting mouth index

--epochs: number of training rounds

--batch_size: training batch size

--output_model: model save path after training
```

(Other main programs are similar)

## Depends on environment

Python 3.8+

PyTorch 1.10+

numpy, librosa, scikit-learn, matplotlib, etc. (see requirements.txt for details)

### Install requirements


```python
pip install -r requirements.txt
```

## Description of results

After training is completed, the script will save the model weight file and training curve graph to facilitate subsequent reasoning and performance analysis.

## Acknowledgments

Thanks to the [VOCASET](https://voca.is.tue.mpg.de/) dataset. Any third-party balances are the property of their respective authors and must be used under their respective licenses.

## 相关工作

- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [VOCA: Voice Operated Character Animation (CVPR 2019)](https://github.com/TimoBolkart/voca)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior, CVPR 2023.](https://github.com/Doubiiu/CodeTalker)
