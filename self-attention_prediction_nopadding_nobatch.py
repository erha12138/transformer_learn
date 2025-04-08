import pandas as pd
import numpy as np
import torch

from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from scipy.special import softmax
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/stock_experiment')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data.iloc[:, 1:]  # 选择第2到第7列

    def __len__(self):
        return len(self.data) - 5 # 要跳过前5个交易日

    def __getitem__(self, idx):
        random_index = np.random.randint(3,6) # 随机选择3-5个交易日 
        # 从 index 开始，取到 index + random_index
        X = np.array(self.data.iloc[idx:idx + random_index, :])
        y = np.array(self.data["Open"].iloc[idx + random_index]) # 下一时刻的开盘价

        return X/100, y/100 # 前3-5个交易日预测下一时刻


class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_q = nn.Linear(6, 6)
        self.W_k = nn.Linear(6, 6)
        self.W_v = nn.Linear(6, 6)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        attention_weight = self.softmax(Q @ K.transpose(-2, -1)) # batch 不转
        attention_score = attention_weight @ V
        return attention_score

class SelfAttentionPredictModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = SelfAttention() # 注意力模块 batch_size, query, 6
        self.fnn = nn.Sequential(
            nn.Linear(6, 32),
            nn.Linear(32, 6)
        )
        self.relu = nn.ReLU()
        self.prediction = nn.Linear(6, 1)

    def forward(self, X):
        H_attn = self.attention(X)
        H_fnn = self.fnn(H_attn)
        H_relu = self.relu(H_fnn)
        H_pooled = H_relu.mean(dim=1) # N 个序列的池化
        y = self.prediction(H_pooled) # batch_size, 1, 1
        return y.squeeze(-1) # 预测的开盘价 batch_size, 1



np.random.seed(42)
torch.random.manual_seed(42)
torch.manual_seed(42)

data = pd.read_csv("MARUTI.csv")

data_selected = data[["Date", "Prev Close", "Open", "High", "Low", "Last", "Close"]]

data_train = data_selected.iloc[:-100]
data_test = data_selected.iloc[-100:]


train_data = StockDataset(data_train)
test_data = StockDataset(data_test)

data_loader = DataLoader(train_data, batch_size=1)
test_data_loader = DataLoader(test_data, batch_size=1)

selfattn_model = SelfAttentionPredictModel().to(device)


writer.add_graph(selfattn_model, input_to_model = torch.rand(1, 5, 6).to(device))


optimizer = AdamW(params=selfattn_model.parameters(), lr=5e-4)
num_epochs = 20
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=num_epochs)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

print("----train start----")
for i_epoch in range(num_epochs):
    total_loss = []
    for batch in data_loader:
        X, y = batch
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        X = X.to(device)
        y = y.to(device)
        y_pred = selfattn_model(X)
        loss = nn.MSELoss()(y_pred, y)

        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    writer.add_scalar('Loss/train', sum(total_loss)/len(total_loss), i_epoch)
    writer.add_scalar('LR', get_lr(optimizer), i_epoch)
    
    print("="*100)
    print("epoch:", i_epoch)
    print("Loss:", sum(total_loss)/ len(total_loss))
    print("LR:", get_lr(optimizer))
    print("="*100)
    scheduler.step()







print("----test start----")
total_loss = []
for X,y in test_data_loader:
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    X = X.to(device)
    y = y.to(device)
    y_pred = selfattn_model(X)
    loss = nn.MSELoss()(y_pred, y)
    total_loss.append(loss.item())
    print("y_pred:", y_pred, "y:", y)
    
print("="*100)
print("test Loss:", sum(total_loss)/ len(total_loss))
print("="*100)


