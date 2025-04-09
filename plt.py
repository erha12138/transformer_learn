import matplotlib.pyplot as plt
import numpy as np

# 定义 .npy 文件的路径
train_path = 'loss_train.npy'
test_path = 'loss_test.npy'

try:
    # 读取 .npy 文件
    data_train = np.load(train_path)
    data_test = np.load(test_path)
    print("成功读取文件。")
except Exception as e:
    print(f"错误: 读取文件时出现问题: {e}")

print(data_train.shape, data_test.shape)

model_name = ["SelfAttentionPredictModel", "Multi_head_SelfAttentionPredictModel_parallel(num_heads=1)",
              "Multi_head_SelfAttentionPredictModel_parallel(num_heads=2)",
              "Multi_head_SelfAttentionPredictModel_parallel(num_heads=3)"]

# 绘制折线图
plt.figure(figsize=(12, 6))
for i in range(data_train.shape[0]):
    plt.plot(data_train[i], label=model_name[i])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存折线图
plt.savefig('result/training_loss_plot.png')
plt.show()

# 绘制柱状图
plt.figure(figsize=(12, 6))
bar_width = 0.2
r = np.arange(data_test.shape[1])
for i in range(data_test.shape[0]):
    plt.bar(r + i * bar_width, data_test[i], width=bar_width, label=model_name[i], alpha=0.5)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Testing Loss over Steps')
plt.xticks([r + bar_width * (data_test.shape[0] - 1) / 2 for r in range(data_test.shape[1])],
           [f'Step {i+1}' for i in range(data_test.shape[1])])
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存柱状图
plt.savefig('result/testing_loss_plot.png')
plt.show()
    