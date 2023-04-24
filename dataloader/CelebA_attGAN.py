import torch
from PIL import Image
import numpy as np

class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataloader, output image and target"""
    
    def __init__(self, key_list, image_feature, target_dict, attr_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.attr_dict = attr_dict 
        self.transform = transform

    def __getitem__(self, index):
        key = self.key_list[index]
        img, target = Image.fromarray(self.image_feature[key][()]), self.target_dict[key]
        attrs = np.array(self.attr_dict[key])

        # print(target.shape)
        # print(attrs.shape)
        # print('dasjlsjkd')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.FloatTensor(target), torch.FloatTensor(attrs)

    def __len__(self):
        return len(self.key_list)