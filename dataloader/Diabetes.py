import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dataloader.Diabetes_data_utils as utils

class DiabetesDataset(Dataset):
    def __init__(self, path, mode, quick_load, bias_name = None, middle_age = 0, training_number = 40000):
        self.bias_name = bias_name
        self.features, self.labels_onehotv, self.bias_onehotv = self.load_data(path, quick_load)
        self.labels = self.onehotvector_to_label(self.labels_onehotv)
        self.bias = self.bias_onehotv
        self.bias_name = bias_name

        # print(self.bias)
        pp, pn, np, nn = utils.split_by_age(self.features, self.labels, self.bias, middle_age)
        minimum = min(len(pp), len(pn), len(np), len(nn))
        print('Before', len(pp), len(pn), len(np), len(nn))
        
        if mode == 'eb1':
            self.features = self.features[pp+nn]
            self.labels = self.labels[pp+nn]
            self.bias = self.bias[pp+nn]
        elif mode == 'eb2':
            self.features = self.features[np+pn]
            self.labels = self.labels[np+pn]
            self.bias = self.bias[np+pn]
        elif mode == 'eb1_balanced':
            self.features = self.features[pp[:minimum]+nn[:minimum]]
            self.labels = self.labels[pp[:minimum]+nn[:minimum]]
            self.bias = self.bias[pp[:minimum]+nn[:minimum]]
        elif mode == 'eb2_balanced':
            self.features = self.features[np[:minimum]+pn[:minimum]]
            self.labels = self.labels[np[:minimum]+pn[:minimum]]
            self.bias = self.bias[np[:minimum]+pn[:minimum]]
        elif mode == 'balanced':
            self.features = self.features[np[:minimum]+pn[:minimum]+pp[:minimum]+nn[:minimum]]
            self.labels = self.labels[np[:minimum]+pn[:minimum]+pp[:minimum]+nn[:minimum]]
            self.bias = self.bias[np[:minimum]+pn[:minimum]+pp[:minimum]+nn[:minimum]]
        

        if mode == 'train':
            self.features = self.features[:training_number]
            self.labels = self.labels[:training_number]
            self.bias = self.bias[:training_number]
        elif mode == 'validation':
            self.features = self.features[training_number:]
            self.labels = self.labels[training_number:]
            self.bias = self.bias[training_number:]
        elif mode == 'all':
            pass

        pp, pn, np, nn = utils.split_by_age(self.features, self.labels, self.bias, middle_age)
        print("After", len(pp), len(pn), len(np), len(nn))

    def __getitem__(self, i):
        feature = self.features[i]
        label = self.labels[i]
        label = np.array(label)
        bias = self.bias[i]
        return feature, label, bias
        # return feature, bias

    def __len__(self):
        data_len = self.features.shape[0]
        return data_len

    def load_data(self, path ,quick_load):
        if quick_load:
            data = utils.quick_load(path)
        else:
            data = utils.data_processing(path)
        data = self.data_preproccess(data)
        features = data[:,:data.shape[1]-2]
        labels = data[:,data.shape[1]-2:]
        bias = utils.get_bias(data, self.bias_name)
        # print(bias)
        return features, labels, bias

    def data_preproccess(self, data):
        data = torch.tensor(data,dtype=torch.float)
        return data

    def onehotvector_to_label(self, onehotvector):
        labels = []
        for v in onehotvector:
            for i in range(v.shape[0]):
                if v[i] != 0:
                    label = i
                    labels.append(label)
                    break
        # labels = torch.tensor(labels, dtype=torch.long).unsqueeze_(1)
        labels = torch.tensor(labels).unsqueeze_(1)
        return labels


if __name__ == '__main__':
    load_path = "./raw_data/adult.csv"
    quick_load_path = "./raw_data/newData.csv"
    dtrain = torch.utils.data.DataLoader(
        AdultDataset(
            path = quick_load_path,
            mode='train',
            quick_load = True,
            bias_name = 'relationship'
        ),
        batch_size=50,
        shuffle=False
    )

    x, y, bias= next(iter(dtrain))
    print(x.shape, x.min(), x.max())
    print(y.shape, y.min(), y.max())

    print('size:', len(dtrain))