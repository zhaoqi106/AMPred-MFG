import matplotlib.pyplot as plt
import os

import numpy as np

# 读取文本文件内容
file_path = 'results/AMES/output.txt'
model_name = file_path.split('/')[-2]
with open(file_path, 'r') as file:
    data = file.read()

# 拆分每个epoch的条目
epoch_entries = data.strip().split('Epoch')

# 绘制auc图像
test_metrics = []
Test_all = []
train_metrics = []
valid_metrics = []
COUNT=0
for entry in epoch_entries:
    lines = entry.split('\n')
    if len(lines) < 2: continue
    true_line = lines[1]
    # print(true_line)
    # print(true_line.split()[0:3])
    # print(true_line.split()[3:6])
    # print(true_line.split()[6:])
    # str = "".join(true_line.split()[0:3])
    # train_metric_value = float(str.split(':')[-1])
    # train_metrics.append(train_metric_value)

    str1 = "".join(true_line.split()[6:])
    test_metric_value = float(str1.split(':')[-1])
    test_metrics.append(test_metric_value)
    COUNT+=1
    if COUNT == 200:
        Test_all.append(test_metrics)
        COUNT=0
        test_metrics = []
    # str2= "".join(true_line.split()[3:6])
    # valid_metrics_value  = float(str2.split(':')[-1])
    # valid_metrics.append(valid_metrics_value)

# 绘制loss图像
# test_loss = []
# train_loss = []
# valid_loss = []
# for entry in epoch_entries:
#     lines = entry.split('\n')
#     if len(lines) < 2: continue
#     true_line = lines[0]
#     test_loss_value = float(true_line.split()[-1].split(':')[-1])
#     test_loss.append(test_loss_value)
#     train_loss_value = float(true_line.split()[-3].split(':')[-1])
#     train_loss.append(train_loss_value)
#     valid_loss_value  = float(true_line.split()[-2].split(':')[-1])
#     valid_loss.append(valid_loss_value)


# 生成epoch序列
epochs = list(range(1, 200 + 1))
average_auc = np.mean(Test_all, axis=0)
# 使用Matplotlib绘图
plt.figure(figsize=(8, 6))
plt.plot(epochs, Test_all[0], label = '1st AUC=0.911', color = 'g')
plt.plot(epochs, Test_all[1], label = '2nd AUC=0.912', color = 'r')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
plt.plot(epochs, Test_all[2], label = '3rd AUC=0.913', color = 'b')
start_value = 0.5671420903304961  # 开始值
end_value = average_auc[-1]    # 结束值
plt.plot([epochs[0], epochs[-1]], [start_value, end_value], color='purple', linestyle='--', label='Average AUC=0.912')

plt.ylabel('AUC', fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.xticks(fontsize=15)  # X轴刻度字体大小
plt.yticks(fontsize=15)  # Y轴刻度字体大小
plt.grid(True)
plt.legend(fontsize=15,loc='lower right')
directory = 'fin_result/figures/'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig('fin_result/figures/' + model_name + '.pdf')

plt.clf()

# plt.plot(epochs, test_loss, label = 'test_auc', color = 'g')
# plt.plot(epochs, train_loss, label = 'train_auc', color = 'r')
# plt.plot(epochs, valid_loss, label = 'valid_auc', color = 'b')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.title('Train & Test Loss')
# plt.grid(True)
# plt.legend()
# plt.savefig('fin_result/figures/' + model_name + '_loss' + '.pdf')
