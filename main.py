from self_attention_prediction import SelfAttentionPredictModel
from multi_head_self_attention_prediction import SelfAttentionPredictModel as Multi_head_SelfAttentionPredictModel_parallel
from dataloader import get_dataloader

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import LinearLR
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

writer = SummaryWriter('runs/main_experiment')

BATCH_SIZE = 8
NUM_EPOCHS = 20

def train(model_name, model, loader, optimizer, scheduler,  criterion, device, num_epochs=NUM_EPOCHS):
    total_loss = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in loader:
            X, y, attention_mask = batch
            X = torch.tensor(X, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            output = model(X, attention_mask)
            loss = criterion(output, y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step()
        writer.add_scalar('Loss/train'+model_name, sum(epoch_loss)/len(epoch_loss), epoch)
        print("="*100)
        print(model_name+"epoch:", epoch)
        print("Loss:", sum(epoch_loss)/len(epoch_loss))
        print("="*100)
        total_loss.append(sum(epoch_loss)/len(epoch_loss))
    
    return total_loss

def test(model_name, model, loader, criterion, device):
    total_loss = []
    for batch in loader:
        X, y, attention_mask = batch
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float32).to(device)

        output = model(X, attention_mask)
        loss = criterion(output, y)
        total_loss.append(loss.item())
    print("="*100)
    print(model_name+"test Loss:", sum(total_loss)/len(total_loss))
    print("="*100)
    return total_loss

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    loader_train, loader_test = get_dataloader(BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model1 = SelfAttentionPredictModel().to(device)
    optimizer1 = optim.AdamW(model1.parameters(), lr=5e-4)
    scheduler1 = LinearLR(optimizer1, start_factor=1.0, end_factor=0.5, total_iters=NUM_EPOCHS)

    model2 = Multi_head_SelfAttentionPredictModel_parallel(num_heads=1).to(device)
    optimizer2 = optim.AdamW(model2.parameters(), lr=5e-4)
    scheduler2 = LinearLR(optimizer2, start_factor=1.0, end_factor=0.5, total_iters=NUM_EPOCHS)
    
    model3 = Multi_head_SelfAttentionPredictModel_parallel(num_heads=2).to(device)
    optimizer3 = optim.AdamW(model3.parameters(), lr=5e-4)
    scheduler3 = LinearLR(optimizer3, start_factor=1.0, end_factor=0.5, total_iters=NUM_EPOCHS)

    model4 = Multi_head_SelfAttentionPredictModel_parallel(num_heads=3).to(device)
    optimizer4 = optim.AdamW(model4.parameters(), lr=5e-4)
    scheduler4 = LinearLR(optimizer4, start_factor=1.0, end_factor=0.5, total_iters=NUM_EPOCHS)

    # para1 = get_param_count(model1)
    # para2 = get_param_count(model2)
    # para3 = get_param_count(model3)
    # para4 = get_param_count(model4)

    model_name = ["SelfAttentionPredictModel", "Multi_head_SelfAttentionPredictModel_parallel(num_heads=1)", "Multi_head_SelfAttentionPredictModel_parallel(num_heads=2)", "Multi_head_SelfAttentionPredictModel_parallel(num_heads=3)"]
    model = [model1, model2, model3, model4]
    optimizer = [optimizer1, optimizer2, optimizer3, optimizer4]
    scheduler = [scheduler1, scheduler2, scheduler3, scheduler4]
    criterion = nn.MSELoss()

    # total_loss = train(model_name[2], model[2], loader_train, optimizer[2], scheduler[2], criterion, device, num_epochs=NUM_EPOCHS)
    # total_loss = train(model_name[3], model[3], loader_train, optimizer[3], scheduler[3], criterion, device, num_epochs=NUM_EPOCHS)

    loss_train = []
    loss_test = []
    for i in range(4):
        total_loss = train(model_name[i], model[i], loader_train, optimizer[i], scheduler[i], criterion, device, num_epochs=NUM_EPOCHS)
        loss_train.append(total_loss)

        total_loss = test(model_name[i], model[i], loader_test, criterion, device)
        loss_test.append(total_loss)
    

    np.save('/loss_train.npy', np.array(loss_train))
    np.save('loss_test.npy', np.array(loss_test))