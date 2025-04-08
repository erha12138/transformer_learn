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


writer = SummaryWriter('runs/stock_experiment_with_padding')
# 这里的 padding 是为了让数据集的长度一致


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PaddedStockDataset(Dataset):
    def __init__(self, data):
        self.data = data.iloc[:, 1:]  # 选择第2到第7列

    def __len__(self):
        return len(self.data) - 5 # 要跳过前5个交易日

    def __getitem__(self, idx):
        random_index = np.random.randint(3,6) # 随机选择3-5个交易日 
        # 从 index 开始，取到 index + random_index
        X = np.array(self.data.iloc[idx:idx + random_index, :])
        y = np.array(self.data["Open"].iloc[idx + random_index]) # 下一时刻的开盘价
        # padding
        if random_index < 5:
            padding = np.zeros((5 - random_index, 6))
            X = np.concatenate((X, padding), axis=0)
            
        attention_mask = np.array([1] * random_index + [0] * (5 - random_index)) # 1 表示有数据，0 表示 padding
        return X/100, y/100, attention_mask # 前3-5个交易日预测下一时刻


class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_q = nn.Linear(6, 6)
        self.W_k = nn.Linear(6, 6)
        self.W_v = nn.Linear(6, 6)
        self.softmax = nn.Softmax(dim=-1)
    
    def masked_softmax(self, attention_score, attention_mask):
        attention_mask1 = attention_mask.unsqueeze(-1).to(device)
        attention_mask2 = attention_mask.unsqueeze(-2).to(device)
        attention_score = attention_score.masked_fill(attention_mask1 == 0, -1e6)
        attention_score = attention_score.masked_fill(attention_mask2 == 0, -1e6)
        attention_weight = self.softmax(attention_score)
        return attention_weight
    
    def forward(self, X, attention_mask):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        attention_score = Q @ K.transpose(-2, -1) # batch 不转
        attention_weight = self.masked_softmax(attention_score, attention_mask) # batch 不转
        
        return attention_weight @ V

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

    def forward(self, X, attention_mask):
        H_attn = self.attention(X, attention_mask)
        H_fnn = self.fnn(H_attn)
        H_relu = self.relu(H_fnn)
        # 忽略attention_mask的pooling
        H_pooled = H_relu * attention_mask.unsqueeze(-1)
        H_pooled = H_pooled.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1) # N 个序列的池化
        y = self.prediction(H_pooled) # batch_size, 1, 1
        return y.squeeze(-1) # 预测的开盘价 batch_size, 1



np.random.seed(42)
torch.random.manual_seed(42)
torch.manual_seed(42)

data = pd.read_csv("MARUTI.csv")

data_selected = data[["Date", "Prev Close", "Open", "High", "Low", "Last", "Close"]]

data_train = data_selected.iloc[:-100]
data_test = data_selected.iloc[-100:]


train_data = PaddedStockDataset(data_train)
test_data = PaddedStockDataset(data_test)

data_loader = DataLoader(train_data, batch_size=8)
test_data_loader = DataLoader(test_data, batch_size=8)


selfattn_model = SelfAttentionPredictModel().to(device)


# writer.add_graph(selfattn_model, input_to_model = (torch.rand(1, 5, 6).to(device), torch.rand(1, 5).to(device)))


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
        X, y, attention_mask = batch
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float32).to(device)
        X = X.to(device)
        y = y.to(device)
        y_pred = selfattn_model(X, attention_mask)
        loss = nn.MSELoss()(y_pred, y)

        total_loss.append(loss.item()/ len(y_pred))
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
for X,y,attention_mask in test_data_loader:
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float32).to(device)
    X = X.to(device)
    y = y.to(device)
    y_pred = selfattn_model(X,attention_mask)
    loss = nn.MSELoss()(y_pred, y)
    total_loss.append(loss.item()/ len(y_pred))
    print("y_pred:", y_pred, "y:", y)
    
print("="*100)
print("test Loss:", sum(total_loss)/ len(total_loss))
print("="*100)


