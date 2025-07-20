import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from  sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import DataLoader, TensorDataset
from ReadData import getData
import gc
import pywt
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
h_dim = 2080
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.bidirectional = False
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True).to(device)
        self.net=nn.Sequential(
            nn.Linear(1080,h_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Linear(h_dim,h_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Linear(h_dim,h_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Linear(h_dim,1080).to(device),
            nn.Sigmoid().to(device)
    )

    def forward(self,x):
        output = self.net(self.lstm(x))
        return output
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.bidirectional = False
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True).to(device)
        self.net=nn.Sequential(
            nn.Linear(1080,h_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Linear(h_dim,h_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Linear(h_dim,h_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Linear(h_dim,1).to(device),
        ).to(device)

    def forward(self,x):
        
    
        output = self.net(self.lstm(x))
        return output

input_dim = 1080 
hidden_dim = 1080 
output_dim = 1080 
batch_size = 8
num_epochs = 1000
learning_rate = 0.005
G = Generator().cuda()
D = Discriminator().cuda()

optim_G = optim.Adam(G.parameters(), lr=9e-4, betas=(0.5,0.9))
optim_D = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5,0.9)) 

criterion = nn.BCEWithLogitsLoss()

txt_dir = r'./data/'
data = getData(txt_dir)
data = np.array(data)

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def wavelet_denoising(data, wavelet='db4', level=3, threshold_factor=0.04):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma
    coeffs_thresh = [pywt.threshold(c, 0.3, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)
dloss = []
gloss = []
for epoch in range(num_epochs):
    all_dloss= 0
    all_gloss= 0
    G.train()
    D.train()
    for real_samples in trans_dataloader:
        data = real_samples["data"].to(device)

        batch_size = data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device) 
        fake_labels = torch.zeros(batch_size, 1).to(device) 
        for _ in range(3):
            predr = D(data)
            z = torch.randn(batch_size,input_dim).cuda()
            xf = G(z).detach()
            predf = D(xf)
            d_real_loss = criterion(predr, real_labels)
            d_fake_loss = criterion(predf, fake_labels)
            loss_D = d_real_loss+d_fake_loss
            all_dloss = all_dloss + loss_D.item()
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
        epochas = 5
        if all_dloss<100:
            epochas = 80
        else:
            epochas = 5
        for _ in range(epochas):
            z = torch.randn(batch_size,input_dim).cuda()
            xf = G(z) # G is updated
            predf = D(xf)
            g_loss = criterion(predf, real_labels)
            all_gloss = all_gloss + g_loss.item()
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()
    dloss.append(all_dloss)
    gloss.append(all_gloss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {all_dloss :.4f}, g_loss: {all_gloss :.4f}')
    if (epoch + 1) % 5 == 0:
        pred_data_list = []
        pred_label_list = []
        for i in range(5):
            print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {all_dloss :.4f}, g_loss: {all_gloss :.4f}')
            G.eval()
            with torch.no_grad():
                batch_size = 8
                z = torch.randn(batch_size,input_dim).cuda()
                generated_samples = G(z)
                predf = nn.Sigmoid()(D(generated_samples))
                pred_label_list.append(predf.cpu().numpy())
                print(predf)
            fig, axs = plt.subplots(2, batch_size, figsize=(15, 6))
            cur_list_data = []
            for i in range(batch_size):
                generated_samples = generated_samples.cpu()
                samples = moving_average(generated_samples[i, :], window_size=40)
                samples = wavelet_denoising(samples)
                samples = moving_average(generated_samples[i, :], window_size=55)
                cur_list_data.append(np.array(samples))
                axs[0, i].plot(samples)
                axs[0, i].set_title(f"Generated {i+1}")
            pred_data_list.append(np.array(cur_list_data))
            idx = np.random.choice(len(data), batch_size, replace=False)
            for i, j in enumerate(idx):
                sample = data[j]
                axs[1, i].plot(sample)
                axs[1, i].set_title(f"Real {i+1}")
            

