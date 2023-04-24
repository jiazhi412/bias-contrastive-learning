import torch
from PIL import Image
import numpy as np
from debias.datasets.utils import TwoCropTransform, get_confusion_matrix

class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataloader, output image and target"""
    
    def __init__(self, key_list, image_feature, target_dict, sex_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.sex_dict = sex_dict
        self.transform = transform

        self.targets = torch.tensor(np.array([target_dict[k] for k in key_list])).long().squeeze()
        self.biases = torch.tensor(np.array([sex_dict[k] for k in key_list])).long()

        # print(self.targets[0])
        # print(type(self.targets[0]))
        # print(self.biases[0])
        # print(type(self.biases[0]))
        # print('dajsla')

        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(num_classes=2,
                                                                                                    targets=self.targets,
                                                                                                    biases=self.biases)


    def __getitem__(self, index):
        key = self.key_list[index]
        img = Image.fromarray(self.image_feature[key][()])
        # target = self.target_dict[key]
        # sex = np.array(self.sex_dict[key])[np.newaxis]
        
        target, sex = self.targets[index], self.biases[index]

        if self.transform is not None:
            img = self.transform(img)

        # return img, torch.FloatTensor(target), torch.FloatTensor(sex), index
        return img, target, sex, index

    def __len__(self):
        return len(self.key_list)