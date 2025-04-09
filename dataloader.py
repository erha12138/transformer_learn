import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

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
        return X, y, attention_mask # 前3-5个交易日预测下一时刻


def get_dataloader(batch_size):
    data = pd.read_csv("MARUTI.csv")

    data_selected = data[["Date", "Prev Close", "Open", "High", "Low", "Last", "Close"]]

    data_train = data_selected.iloc[:-100]
    data_test = data_selected.iloc[-100:]

    train_data = PaddedStockDataset(data_train)
    test_data = PaddedStockDataset(data_test)

    data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    return data_loader, test_data_loader