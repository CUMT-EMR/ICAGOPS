import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader,Dataset
from  sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
from ReadDataCNN import getData
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
seq_len = 1024
batch_size = 32
model_name = "BPNN"

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(seq_len,seq_len*2).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.linear2 = nn.Linear(seq_len*2,seq_len).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.linear3 = nn.Linear(seq_len,1).to(device)
    def forward(self,data):
        data = torch.unsqueeze(data,dim=1)
        tgt = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(data)))))
        return torch.squeeze(tgt)
    

class DataHandler(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        data = data.reset_index()
        self.EMR=data['EMR']
        self.AE=data['AE']
        self.GAS=data['GAS']
        self.EMR_pad=data['EMR_pad']
        self.AE_pad=data['AE_pad']
        self.GAS_pad=data['GAS_pad']
        self.EMR_label=data['EMR_label']
        self.AE_label=data['AE_label']
        self.GAS_label=data['GAS_label']
        self.EMR_len=data['EMR_len']
        self.AE_len=data['AE_len']
        self.GAS_len=data['GAS_len']
    def __getitem__(self, index):
        return {
                "EMR_pad":torch.tensor(self.EMR_pad[index],dtype=torch.float),
                "AE_pad":torch.tensor(self.AE_pad[index],dtype=torch.float),
                "GAS_pad":torch.tensor(self.GAS_pad[index],dtype=torch.float),
                "EMR_len":torch.tensor(self.EMR_len[index],dtype=torch.float),
                "AE_len":torch.tensor(self.AE_len[index],dtype=torch.float),
                "GAS_len":torch.tensor(self.GAS_len[index],dtype=torch.float),
                "EMR_label":torch.tensor(self.EMR_label[index],dtype=torch.float),
                "AE_label":torch.tensor(self.AE_label[index],dtype=torch.float),
                "GAS_label":torch.tensor(self.GAS_label[index],dtype=torch.float)
                }
        
        
    def __len__(self):
        return len(self.EMR)
prediction_vali_loss = []
prediction_train_loss = []
criterion = nn.BCEWithLogitsLoss().to(device)
def train(trans_dataloader,test_validation,iter):
    model = Model()
    optim_fnc = optim.Adam(model.parameters(),lr=1e-4,weight_decay=0.001)
    scheduler_1 = StepLR(optim_fnc, step_size=500, gamma=0.8)
    criterion = nn.BCEWithLogitsLoss().to(device)
    print('trans_dataloader',len(trans_dataloader))
    loss_data=[]
    opti_rate =   []
    opti_rate.append(optim_fnc.defaults['lr'])
    model.train()
    Echop =800
    for echop in range(Echop):
        model.train()
        total_loss = 0
        items = 0
        with torch.enable_grad():
            for data_item in trans_dataloader:
                EMR_pad = data_item["EMR_pad"].to(device)
                AE_pad = data_item["AE_pad"].to(device)
                GAS_pad = data_item["GAS_pad"].to(device)
                EMR_label = data_item["EMR_label"].to(device)
                AE_label = data_item["AE_label"].to(device)
                GAS_label = data_item["GAS_label"].to(device)
                preduction = model(AE_pad).squeeze().to(device)
                loss_val = criterion(preduction,AE_label).to(device)
                optim_fnc.zero_grad()
                loss_val.backward()
                items+=1 
                optim_fnc.step()
                total_loss+=loss_val.item()
            loss_data.append(total_loss)
            opti_rate.append(optim_fnc.param_groups[0]['lr'])
            loss_data_rate = deepcopy(total_loss)
            scheduler_1.step(loss_data_rate)
            print('Epoch {},  Totle Loss {}'.format(echop, total_loss) )
            prediction(test_validation,model,echop,'test_validation',iter)
            prediction(trans_dataloader,model,echop,'trans_dataloader',iter)
        if echop%10==0:
            torch.save(model,'./model/resnet_ae_model_'+str(echop)+".pt")
    result = pd.DataFrame({"ecoph":np.arange(0,Echop),"loss":loss_data})
    result_vali_loss = pd.DataFrame({"ecoph":np.arange(0,Echop),"loss":prediction_vali_loss})
    result_train_loss = pd.DataFrame({"ecoph":np.arange(0,Echop),"loss":prediction_train_loss})
    result_opti = pd.DataFrame({"ecoph":np.arange(0,Echop+1),"opti_step":opti_rate})
    data_result_write = pd.ExcelWriter("./trainlossAE/{}_{}_result_loss_BPNN.xlsx".format(model_name,iter))
    data_testvali_result_write = pd.ExcelWriter("./trainlossAE/{}_{}_result_vali_loss_BPNN.xlsx".format(model_name,iter))
    data_trainvail_result_write = pd.ExcelWriter("./trainlossAE/{}_{}_result_train_loss_BPNN.xlsx".format(model_name,iter))
    data_optostep_result_write = pd.ExcelWriter("./trainlossAE/{}_{}_result_train_opti_step_BPNN.xlsx".format(model_name,iter))
    result.to_excel(data_result_write)
    result_vali_loss.to_excel(data_testvali_result_write)
    result_train_loss.to_excel(data_trainvail_result_write)
    result_opti.to_excel(data_optostep_result_write)
    data_result_write.close()
    data_testvali_result_write.close()
    data_trainvail_result_write.close()
    data_optostep_result_write.close()
    plt.plot(np.arange(0,Echop),loss_data)
    print(loss_data)
    plt.show()
    
def prediction(test_loader,model,echop,name,iter):
    wasi_predictions,fenchen_predictions,yali_predictions,co_predictions,fenchen_target,wasi_target,fenchen_true_val,wasi_true_val,yali_true_val,co_true_val,yali_target,co_target = [],[],[],[],[],[],[],[],[],[],[],[]
    total_loss = 0
    with torch.no_grad():
        model.eval()
        items = 0
        for data_item in test_loader:
            AE_pad = data_item["AE_pad"].to(device)
            AE_label = data_item["AE_label"].to(device)
            pred = model(AE_pad)
            preduction = torch.sigmoid(pred.squeeze().to(device))
            wasi_predictions.extend(preduction)
            wasi_target.extend(AE_label)
            loss_val = criterion(pred,AE_label).to(device)
            items+=1
            total_loss+=loss_val.item()
        if name =='test_validation':
            prediction_vali_loss.append(total_loss)
            print('prediction_vali_loss Totle Loss {}'.format(total_loss) )
        elif name =='trans_dataloader':
            prediction_train_loss.append(total_loss)
            print('prediction_train_loss Totle Loss {}'.format(total_loss) )
    result = pd.DataFrame({"wasi_predictions":torch.Tensor(wasi_predictions).cpu(),
                           "wasi_target":torch.Tensor(wasi_target).cpu()})
    data_result_write = pd.ExcelWriter('./tmpBPNNAE/{}_{}_'.format(model_name,iter)+name+"_result_test_new_"+str(echop)+"_BPNN.xlsx")
    result.to_excel(data_result_write)
    data_result_write.close()
scaler = MinMaxScaler() 
def main():
    txt_dir = r'./data/alldata/'
    EMR_feature_data,AE_feature_data,GAS_feature_data,EMR_feature_pad_data,AE_feature_pad_data,GAS_feature_pad_data,EMR_label_data,AE_label_data,GAS_label_data,EMR_feature_data_len,AE_feature_data_len,GAS_feature_data_len = getData(txt_dir)
    data_pd = pd.DataFrame({"EMR":EMR_feature_data,"AE":AE_feature_data,"GAS":GAS_feature_data,"EMR_pad":EMR_feature_pad_data,
              "AE_pad":AE_feature_pad_data,"GAS_pad":GAS_feature_pad_data,"EMR_label":EMR_label_data
              ,"AE_label":AE_label_data,"GAS_label":GAS_label_data,"EMR_len":EMR_feature_data_len,
              "AE_len":AE_feature_data_len,"GAS_len":GAS_feature_data_len})
    trans_data,test_validation = train_test_split(data_pd,test_size=0.2,train_size=0.8,shuffle=True,random_state=1)
    test_data,validation_data = train_test_split(test_validation,test_size=0.5,train_size=0.5,shuffle=True,random_state=1)
    trans_dataloader = DataLoader(DataHandler(trans_data),batch_size=batch_size,shuffle=True)
    validation_dataloader = DataLoader(DataHandler(validation_data),batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(DataHandler(test_validation),batch_size=batch_size,shuffle=True)
    return trans_dataloader,validation_dataloader,test_dataloader,test_validation
    
if __name__=='__main__':
    for i in range(10):
        prediction_vali_loss = []
        prediction_train_loss = []
        trans_dataloader,validation_dataloader,test_dataloader,test_validation = main()
        train(trans_dataloader,test_dataloader,iter=i)
    
