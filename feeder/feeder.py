import torch
import numpy as np


class Feeder(torch.utils.data.Dataset):
    """ Feeder for data inputs """

    def __init__(self, data_path, label_path=None, selected_features=None):
        self.data_path = data_path
        self.label_path = label_path
        self.selected_features = selected_features
        self.load_data()

    def load_data(self):
        data = np.load(self.data_path, mmap_mode='r')
        print(f"原始数据的尺寸: {data.shape}")

        if data.ndim == 4:
            # 处理四维数据
            self.data = data.transpose(0, 2, 1, 3).reshape(-1, data.shape[1], data.shape[3])
            # 确保第一个维度为 9
            self.data = self.data.transpose(1, 0, 2)
        elif data.ndim == 3:
            self.data = data
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")

        if self.selected_features is not None:
            self.data = self.data[:, :, self.selected_features]

        print(f"处理后数据的尺寸: {self.data.shape}")

        if self.label_path:
            self.label = np.load(self.label_path, mmap_mode='r')
            print(f"标签数据的尺寸: {self.label.shape}")
        else:
            self.label = None

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        if self.label is not None:
            if self.data.ndim == 4:
                original_index = index % len(self.label)
                return self.data[:, index, :], self.label[original_index]
            elif self.data.ndim == 3:
                original_index = index % len(self.label)
                return self.data[:, index, :], self.label[original_index]
        else:
            return self.data[:, index, :]