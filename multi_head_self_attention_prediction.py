from dataloader import get_dataloader
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from scipy.special import softmax

BATCH_SIZE = 8

loader_train, loader_test = get_dataloader(BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

class Multi_head_SelfAttentionPredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(6,6) / 100)
        self.W_k = nn.Parameter(torch.randn(6,6) / 100)
        self.W_v = nn.Parameter(torch.randn(6,6) / 100)
        # 注意力头1
        self.W_q_1 = nn.Parameter(torch.randn(6,6) / 100)
        self.W_k_1 = nn.Parameter(torch.randn(6,6) / 100)
        self.W_v_1 = nn.Parameter(torch.randn(6,6) / 100)
        # 注意力头2
        self.W_q_2 = nn.Parameter(torch.randn(6,6) / 100)
        self.W_k_2 = nn.Parameter(torch.randn(6,6) / 100)
        self.W_v_2 = nn.Parameter(torch.randn(6,6) / 100)

        self.W_o = nn.Parameter(torch.randn(12,6) / 100)

        self.softmax = nn.Softmax(-1)
    
    def forward(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        Q_1 = Q @ self.W_k_1
        K_1 = K @ self.W_k_1
        V_1 = V @ self.W_v_1
        attention_score_1 = Q_1 @ K_1.transpose(-2, -1)
        attention_weight_1 = self.softmax(attention_score_1)
        attention_output_1 = attention_weight_1 @ V_1

        Q_2 = Q @ self.W_k_2
        K_2 = K @ self.W_k_2
        V_2 = V @ self.W_v_2
        attention_score_2 = Q_2 @ K_2.transpose(-2, -1)
        attention_weight_2 = self.softmax(attention_score_2)
        attention_output_2 = attention_weight_2 @ V_2
        attention_output = torch.cat((attention_output_1, attention_output_2), dim=-1)
        attention_output = attention_output @ self.W_o
        return attention_output



class Multi_head_SelfAttentionPredictModel_2(nn.Module):
    def __init__(self, hidden_size=6, num_heads=2):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.W_k = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.W_v = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)

        self.W_q_m = [nn.Parameter(torch.randn(hidden_size, hidden_size//num_heads) / 100) for _ in range(num_heads)]
        self.W_k_m = [nn.Parameter(torch.randn(hidden_size, hidden_size//num_heads) / 100) for _ in range(num_heads)]
        self.W_v_m = [nn.Parameter(torch.randn(hidden_size, hidden_size//num_heads) / 100) for _ in range(num_heads)]
        self.W_o = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.softmax = nn.Softmax(-1)


    def forward(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        attention_output_list = []
        for i in range(len(self.W_q_m)):
            Q_m = Q @ self.W_q_m[i]
            K_m = K @ self.W_k_m[i]
            V_m = V @ self.W_v_m[i]
            attention_score_m = Q_m @ K_m.transpose(-2, -1)
            attention_weight_m = self.softmax(attention_score_m)
            attention_output_m = attention_weight_m @ V_m
            attention_output_list.append(attention_output_m)

        attention_output = torch.cat(attention_output_list, dim=-1)
        attention_output = attention_output @ self.W_o
        return attention_output


class Multi_head_SelfAttentionPredictModel_parallel(nn.Module):
    def __init__(self, hidden_size=6, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.W_q = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.W_k = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.W_v = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.W_q_m = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100) # -1 : d/m * m
        self.W_k_m = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100) # -1 : d/m * m
        self.W_v_m = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100) # -1 : d/m * m
        self.W_o = nn.Parameter(torch.randn(hidden_size, hidden_size) / 100)
        self.softmax = nn.Softmax(-1)
    
    def masked_softmax(self, attention_score, attention_mask):
        attention_mask1 = attention_mask.unsqueeze(-1).unsqueeze(1)
        attention_mask2 = attention_mask.unsqueeze(-2).unsqueeze(1)
        attention_score = attention_score.masked_fill(attention_mask1 == 0, -1e6)
        attention_score = attention_score.masked_fill(attention_mask2 == 0, -1e6)
        attention_weight = self.softmax(attention_score)
        return attention_weight


    def multi_head_attention(self, Q, K, V, attention_mask):
        Q_stack = Q.reshape(Q.size(0), self.num_heads, -1, self.hidden_size // self.num_heads) # 拆开 B, number_of_heads, n, d/m
        K_stack = K.reshape(K.size(0), self.num_heads, -1, self.hidden_size // self.num_heads)
        V_stack = V.reshape(V.size(0), self.num_heads, -1, self.hidden_size // self.num_heads)
        attention_score = Q_stack @ K_stack.transpose(-2, -1)
        attention_weight = self.masked_softmax(attention_score, attention_mask)
        attention_output = attention_weight @ V_stack # B, number_of_heads, n, d/m
        attention_output = attention_output.reshape(attention_output.size(0), -1, self.hidden_size)
        return attention_output
    
    def forward(self, X, attention_mask=None):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        Q_m = Q @ self.W_q_m # B, n , d/m * m
        K_m = K @ self.W_k_m
        V_m = V @ self.W_v_m

        attention_output = self.multi_head_attention(Q_m, K_m, V_m, attention_mask) # B, n , d/m * m
        attention_output = attention_output @ self.W_o

        return attention_output


class SelfAttentionPredictModel(nn.Module):
    def __init__(self, hidden_size=6, num_heads=2):
        super().__init__()
        self.attention = Multi_head_SelfAttentionPredictModel_parallel(hidden_size=6, num_heads=2) # 注意力模块 batch_size, query, 6
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

if __name__ == "__main__":
    # 这里是测试代码


    attention = Multi_head_SelfAttentionPredictModel()

    attention2 = Multi_head_SelfAttentionPredictModel_2()

    attention3 = Multi_head_SelfAttentionPredictModel_parallel()
    input = torch.randn(2, 5, 6)
    output1 = attention(input)
    output2 = attention2(input)
    ouput3 = attention3(input, torch.tensor([[1, 1, 1, 0, 0],[1,1,1,1,0]]).unsqueeze(0))
    print(output1.shape)
    print(output2.shape)
    print(ouput3.shape)
