import torch

# 创建一个张量
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6],[7, 8, 9]],
                       [[10, 11, 12], [13, 14, 15],[16, 17, 18]]])
# 创建一个可以广播的布尔掩码
mask1 = torch.tensor([[1,1,0],[1,0,0]]).unsqueeze(-1)
mask2 = torch.tensor([[1,1,0],[1,0,0]]).unsqueeze(-2)
# 使用 masked_fill 函数进行替换
result = tensor.masked_fill(mask1==0, -1)
result = result.masked_fill(mask2==0, -1)
print("原始张量:")
print(tensor)
print("掩码:")
print(mask1,mask2)
print("替换后的张量:")
print(result)

a = torch.tensor([1])
print(-1e6 * 0)