import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import DataLoader, TensorDataset
from ReadData_diffusion import getData
import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Unet1D(
    dim = 512,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 1080,
    timesteps = 1000,
    objective = 'pred_v'
).to(device)
txt_dir = r'./data/'
data = getData(txt_dir)
scaler = MinMaxScaler(feature_range=(-1, 1))
data = np.array(data)
tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
dataset = Dataset1D(tensor_data)
trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 16,
    train_lr = 8e-5,
    train_num_steps = 700000,
    gradient_accumulate_every = 2,  
    ema_decay = 0.995,              
    amp = True,                 
)
trainer.train()
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def wavelet_denoising(data, wavelet='db4', level=3, threshold_factor=0.04):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  
    threshold = threshold_factor * sigma 
    coeffs_thresh = [pywt.threshold(c, 0.4, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)
for epoch in range(30):
    cur_list_data = []
    cur_list_data_nodenoise = []
    sampled_seq = diffusion.sample(batch_size=5).cpu()
    for i in range(5):
        generated_samples = sampled_seq.squeeze().cpu()
        samples = moving_average(generated_samples[i, :], window_size=50)
        samples = wavelet_denoising(samples)
        cur_list_data.append(np.array(samples))
        cur_list_data_nodenoise.append(np.array(generated_samples[i, :]))
    df_data = pd.DataFrame(cur_list_data).T
    df_data_nodenoise = pd.DataFrame(cur_list_data_nodenoise).T
    df = pd.concat([df_data], axis=1)
    df_nodenoise = pd.concat([df_data_nodenoise], axis=1)
sampled_seq = diffusion.sample(batch_size=4).to(device)
batch_size = 4
pred_data_list = []
fig, axs = plt.subplots(2, batch_size, figsize=(15, 6))
cur_list_data = []
for i in range(batch_size):
    generated_samples = sampled_seq.squeeze().cpu()
    samples = moving_average(generated_samples[i, :], window_size=50)
    samples = wavelet_denoising(samples)
    cur_list_data.append(np.array(samples))
    axs[0, i].plot(samples)
    axs[0, i].set_title(f"Generated {i+1}")
pred_data_list.append(np.array(cur_list_data))
idx = np.random.choice(len(data), batch_size, replace=False)
for i, j in enumerate(idx):
    sample = data[j]
    axs[1, i].plot(sample)
    axs[1, i].set_title(f"Real {i+1}")
plt.tight_layout()
plt.show()
