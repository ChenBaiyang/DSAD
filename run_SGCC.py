import torch, time
import torch.optim as optim
import numpy as np
from DSAD import DataLoader, SiameseAR
from sklearn.metrics import roc_auc_score, average_precision_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
sub_seq_len = 7
dataloader = DataLoader(sub_seq_len=sub_seq_len, sub_seq_step=sub_seq_len,
                        batch_size=batch_size, mode='week')
y = torch.from_numpy(dataloader.labels).float()
X = torch.from_numpy(dataloader.data).float().to(device)
X -= X.mean(dim=1, keepdims=True)
mask = torch.from_numpy(dataloader.mask).float().to(device)

# 创建模型实例
hidden_dim = 16
num_heads = 4
num_layers = 2

model = SiameseAR(sub_seq_len, hidden_dim, num_layers, num_heads)
model.to(device)

# 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练阶段
num_epochs = 3
print(X.shape)
print(sub_seq_len, hidden_dim, num_layers, num_heads)
all_idx = np.arange(dataloader.n_sample)
set1, set2 = all_idx, all_idx
for epoch in range(num_epochs):
    total_loss = 0.0
    for idx1, idx2 in dataloader.batch_two_parts_idx(set1, set2):  # 假设您有一个数据加载器用于加载时间序列数据
        optimizer.zero_grad()

        # 前向传播
        inputs1 = X[idx1]  # 获取输入数据
        input_mask1 = mask[idx1]
        inputs2 = X[idx2]  # 获取输入数据
        input_mask2 = mask[idx2]
        l1, l2 = model(inputs1, input_mask1, inputs2, input_mask2)
        reconstruction_loss = l1.mean() + l2.mean()
        # 累加或平均重构损失
        total_loss += reconstruction_loss.item()

        # 反向传播和优化
        reconstruction_loss.backward()
        optimizer.step()

    # 打印每个epoch的重构损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Reconstruction Loss: {total_loss}")

    # Evaluation
    scores_1 = []
    scores_2 = []
    model.eval()
    with torch.no_grad():
        for idx in dataloader.batch_all_idx_origin():
            inputs = X[idx]
            input_mask = mask[idx]
            l1, l2 = model(inputs, input_mask, inputs, input_mask)
            l1 = l1.cpu().numpy()
            l2 = l2.cpu().numpy()
            score_1 = l1.mean(axis=2).max(axis=1)
            score_2 = l2.mean(axis=2).max(axis=1)
            scores_1.append(score_1)
            scores_2.append(score_2)

    scores_1 = np.concatenate(scores_1, axis=0)
    scores_2 = np.concatenate(scores_2, axis=0)


    scores_1_order = np.argsort(scores_1)[int(dataloader.n_sample*0.9):]
    scores_2_order = np.argsort(scores_2)[int(dataloader.n_sample*0.9):]
    pred_anomaly = np.intersect1d(scores_1_order, scores_2_order)

    set1 = np.setdiff1d(all_idx, pred_anomaly)
    set2 = set1

    ap = average_precision_score(y, scores_1 + scores_2)
    auc = roc_auc_score(y, scores_1 + scores_2)
    print("\t{}, ap: {:.4f}, auc: {:.4f}"
          .format(time.ctime()[4:-5], ap, auc))

    model.train()
    np.random.shuffle(set1)
    np.random.shuffle(set2)
    print(len(scores_1_order), len(scores_2_order), len(pred_anomaly), len(set1))

    scores = ['SGCC', "DSAD", str(epoch), str(hidden_dim), str(sub_seq_len), str(auc)[:8], str(ap)[:8]]
    open('results_DSAD.csv', 'a').write(','.join(scores) + '\n')

