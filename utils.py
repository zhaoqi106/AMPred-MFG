import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score, recall_score, f1_score
import dgl
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def AUC(tesYAll, tesPredictAll):
    tesAUC = roc_auc_score(tesYAll, tesPredictAll)
    tesAUPR = average_precision_score(tesYAll, tesPredictAll)
    return tesAUC,tesAUPR
def RMSE(tesYAll,tesPredictAll):
    return mean_squared_error(tesYAll, tesPredictAll,squared=False),0
def confusion_matrix1(true_y,PED):
    TN, FP, FN, TP = confusion_matrix(true_y,PED).ravel()
    return TN, FP, FN, TP
class GraphDataset_Classification(Dataset):
    def __init__(self, g_list, y_tensor,fp_list,macc_list,ecfp_list,x_list,a_list):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.fp_list = fp_list
        self.macc_list = macc_list
        self.ecfp_list = ecfp_list
        self.x_list = x_list
        self.a_list = a_list
        self.len = len(g_list)


    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx],self.fp_list[idx],self.macc_list[idx],self.ecfp_list[idx],self.x_list[idx],self.a_list[idx]

        # return self.g_list[idx], self.y_tensor[idx], self.ecfp_list[idx]

    def __len__(self):
        return self.len


class GraphDataLoader_Classification(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Classification, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        #batched_ws = torch.stack([item[2] for item in batch])
        batch_fp = torch.stack([item[2] for item in batch])
        batch_macc = torch.stack([item[3] for item in batch])
        batch_ecfp = torch.stack([item[4] for item in batch])
        batch_X = torch.stack([item[5] for item in batch])
        batch_A = torch.stack([item[6] for item in batch])
        return (batched_gs, batched_ys,batch_macc,batch_fp,batch_ecfp ,batch_X,batch_A)

class GraphDataset_Regression(Dataset):
    def __init__(self, g_list, y_tensor):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx]
    def __len__(self):
        return self.len

class GraphDataLoader_Regression(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Regression, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        return (batched_gs, batched_ys)

def tsen(epoch,out_all,traYAll):
    if  epoch == 199:
        out = torch.tensor(out_all)

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # 假设 X 和 y 已经定义
        X = out  # 特征数据
        y = traYAll  # 标签meidai

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA 预处理
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        # 自定义颜色
        colors = np.array(['', ''])  # 在这里定义自定义颜色
        colors = ['#d86967', '#58539f']
        cmap_custom = ListedColormap(colors)
        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)

        # 可视化 t-SNE 结果
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap_custom, s=25)
        # ,edgecolors='#778899'
        ax = plt.gca()  # 获取当前坐标轴
        ax.spines['top'].set_visible(True)  # 隐藏上边框
        ax.spines['right'].set_visible(True)  # 隐藏右边框
        ax.spines['right'].set_color('#DCDCDC')  # 设置左边框颜色
        ax.spines['right'].set_linewidth(1)  # 设置左边框宽度
        ax.spines['top'].set_color('#DCDCDC')  # 设置下边框颜色
        ax.spines['top'].set_linewidth(1)  # 设置下边框宽度
        ax.spines['left'].set_color('#DCDCDC')  # 设置左边框颜色
        ax.spines['left'].set_linewidth(1)  # 设置左边框宽度
        ax.spines['bottom'].set_color('#DCDCDC')  # 设置下边框颜色
        ax.spines['bottom'].set_linewidth(1)  # 设置下边框宽度
        plt.title("")
        plt.xlabel("t-SNE-0",fontsize=28)
        plt.ylabel("t-SNE-1",fontsize=28)

        plt.xticks([])
        plt.yticks([])
        handles, _ = scatter.legend_elements()
        legend_labels = ['Mutagens','Non-Mutagens']
        plt.legend(handles=handles, labels=legend_labels,fontsize=18,loc='upper left',handletextpad=0,borderpad=0.2)

        plt.show()