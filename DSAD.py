import torch, os
import torch.nn as nn
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader:
    def __init__(self, sub_seq_len=7, sub_seq_step=7,
                 batch_size=128, train_rate=0.8, use_cache=False, mode='week'):
        self.sub_seq_len = sub_seq_len
        self.sub_seq_step = sub_seq_step
        self.use_cache = use_cache
        self.batch_size = batch_size

        data_file_name = 'dataset/SGCC/SGCC_with_mask.npz'
        if os.path.exists(data_file_name):
            file = np.load(data_file_name)
            self.data = file['data']
            self.labels = file['label']
            self.mask = file['mask']
        else:
            self.load_data()

        self.data = np.concatenate([self.data, self.mask], axis=-1)
        self.data = self.get_sub_seq()
        self.data, self.mask = self.data[:,:,:,0],self.data[:,:,:,1]
        self.apply_mask2input(mode=mode)
        self.n_sample, self.seq_len, self.n_feature = self.data.shape
        self.all_idx = np.arange(len(self.data))

    def load_data(self):
        from datetime import datetime, timedelta
        if os.path.exists('dataset/SGCC/data.csv'):
            file = pd.read_csv('dataset/SGCC/data.csv')
        elif os.path.exists('dataset/SGCC/data.zip'):
            from zipfile import ZipFile
            csv = ZipFile('dataset/SGCC/data.zip')
            file = csv.open("data.csv")
            file = pd.read_csv(file)

        # 确保少数类被标记为异常类，用1表示，正常类用0表示
        labels = file.values[:, 1].astype(np.int16)         # 第2列为标记
        label_list, label_counts = np.unique(labels, return_counts=True)
        pos_label = label_list[np.argmin(label_counts)]
        self.labels = (labels == pos_label)[:, np.newaxis]

        file = file.drop(['CONS_NO', 'FLAG'], axis=1)       # 去除多余的第1、2列
        date = np.array([datetime.strptime(day, '%Y/%m/%d') for day in file.columns])
        new_order = date.argsort()                          # 所有列按照日期先后重新排序
        file = file.iloc[:, new_order]
        date = date[new_order]

        ### 所有列按照周对齐，如果连续两天之间差异不是一天或者一周，则必定有一天的数据缺失；缺少的填补为空白值
        weekday = np.array([datetime.date(day).weekday() for day in date])
        for idx in range(len(weekday) - 1):
            if weekday[idx+1] - weekday[idx] != 1 and weekday[idx+1] - weekday[idx] != -6:
                new_column_name = datetime.strftime(date[idx] + timedelta(days=1), '%Y/%m/%d')
                file.insert(idx+1, new_column_name, None)
                print("\tFilling a blank date:", new_column_name)

        # 提取缺失值特征
        self.mask = np.logical_not(file.isna().values)[:, :, np.newaxis]

        # 缺失值填充为0
        x_fill0 = file.fillna(0).values.astype(np.float32)
        self.data = x_fill0[:, :, np.newaxis]
        idx = np.arange(len(self.data))
        np.random.seed(0)
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        self.mask = self.mask[idx]
        np.savez('dataset/SGCC/SGCC_with_mask',
                 data=self.data, label=self.labels, mask=self.mask)

    def get_sub_seq(self):
        n, m, _ = self.data.shape
        data_sub_seq = []
        for i in range(m):
            end = i * self.sub_seq_step + self.sub_seq_len
            if end > m:
                break
            else:
                start = i * self.sub_seq_step
                temp = self.data[:, start:end]
                data_sub_seq.append(temp)
        return np.concatenate([data_sub_seq]).transpose(1, 0, 2, 3)

    def batch_two_parts_idx(self, neg, pos):
        for i in range(len(neg) // self.batch_size + 1):
            idx1 = neg[i * self.batch_size:(i + 1) * self.batch_size]
            len_pos = len(pos)
            pos_start, pos_end = i * self.batch_size, (i + 1) * self.batch_size
            if pos_start > len_pos:
                pos_start = pos_start % len_pos
                pos_end = pos_start + self.batch_size

            idx2 = pos[pos_start: pos_end]
            if len(idx1) > 0:
                yield idx1, idx2

    def batch_all_idx_origin(self):
        for i in range(len(self.data) // self.batch_size + 1):
            idx = np.arange(len(self.data))[i * self.batch_size:(i + 1) * self.batch_size]
            if len(idx) > 0:
                yield idx

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def apply_mask2input(self, mode='week'):
        if mode == 'week':
            print('\tApplying masks to input data by week')
            self.mask = np.where(self.mask.sum(axis=-1, keepdims=True) > self.sub_seq_len - 1e-5, 1, 0)
            self.data = self.data * self.mask
        else:
            print('\tApplying masks to input data by day')
            self.data = self.data * self.mask


class SiameseAR(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=100, num_layers=4, num_heads=4):
        super(SiameseAR, self).__init__()

        self.shared_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)

        self.encoder_layers1 = nn.TransformerEncoderLayer(hidden_dim, num_heads//2, hidden_dim * 2, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layers1, num_layers//2)
        self.decoder_layers1 = nn.TransformerDecoderLayer(hidden_dim, num_heads//2, hidden_dim * 2, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(self.decoder_layers1, num_layers//2)
        self.recon = nn.Linear(hidden_dim, input_dim)
        self.recon1 = nn.Linear(hidden_dim, input_dim)


        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x, mask, x1, mask1):
        embedded = self.shared_embedding(x)
        embedded_with_positions = self.positional_encoding(embedded)

        encoder_output = self.encoder(embedded_with_positions)
        reconstructed_output = self.decoder(embedded_with_positions, encoder_output)
        reconstructed_output = self.recon(reconstructed_output)
        loss = self.criterion(x, reconstructed_output) * mask

        embedded1 = self.shared_embedding(x1)
        embedded_with_positions1 = self.positional_encoding(embedded1)

        encoder_output1 = self.encoder1(embedded_with_positions1)
        reconstructed_output1 = self.decoder1(embedded_with_positions1, encoder_output1)
        reconstructed_output1 = self.recon1(reconstructed_output1)
        loss1 = self.criterion(x1, reconstructed_output1) * mask1

        return loss, loss1


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionalEncoding, self).__init__()

        max_seq_len = 2000  # Adjusted sequence length
        position_encoding = torch.zeros(max_seq_len, hidden_dim)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        position_encoding[:, 0::2] = torch.sin(positions * div_term)
        position_encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        seq_len = x.shape[1]
        x = x + self.position_encoding[:seq_len, :]
        return x

