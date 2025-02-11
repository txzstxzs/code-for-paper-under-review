import numpy as np
import torch


def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        data = data[::-1]                          # 逆序输出

        norm_data = normalize(data)
        
        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)

        '打乱样本顺序  实验证明意义不大 因为后面其他操作还会打乱一次样本 只需要一次打乱的操作就行'
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
            
    '这个函数表示使用TimeDataset[i]时返回的值'
    def __getitem__(self, index):
        return self.samples[index]
    
    '这个函数表示使用len(TimeDataset)时返回的值'
    def __len__(self):
        return len(self.samples)
