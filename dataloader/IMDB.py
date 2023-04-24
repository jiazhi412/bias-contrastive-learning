import torch
from PIL import Image
import numpy as np
import os

class IMDBDataset(torch.utils.data.Dataset):
    """CelebA dataloader, output image and target"""
    
    def __init__(self, image_feature, age_dict, sex_dict, mode, eb1, eb2, test, dev_or_test, transform=None):
        self.image_feature = image_feature
        self.age_dict = age_dict
        self.sex_dict = sex_dict
        self.transform = transform
        self.mode = mode

        if mode.startswith('eb1'):
            self.key_list = eb1
        elif mode.startswith('eb2'):
            self.key_list = eb2
        elif mode.startswith('unbiased'):
            self.key_list = test
        elif mode.startswith('all'):
            self.key_list = eb1 + eb2 + test
        
        # TODO IMDB_split_by_sex variance
        print(f'[{mode}] {dev_or_test}:')
        if mode.startswith('unbiased_test'):
            df = IMDB_split_by_sex(self.age_dict, self.sex_dict, self.key_list)
            self.key_list = new_dataset_test_eb(age_dict, sex_dict, test, df, mode)
            df = IMDB_split_by_sex(self.age_dict, self.sex_dict, self.key_list)
        else:
            df = IMDB_split_by_sex(self.age_dict, self.sex_dict, self.key_list)

        csv_name = f'/nas/home/jiazli/code/Adversarial-Filter-Debiasing/{mode}.csv'
        if not os.path.exists(csv_name):
            df.to_csv(csv_name,index=False)
            
        if dev_or_test == 'train':
            pass
        elif dev_or_test == 'dev':
            self.key_list = self.key_list[:len(self.key_list) // 10]
        elif dev_or_test == 'test':
            self.key_list = self.key_list[len(self.key_list) // 10:]
        elif dev_or_test == 'dev_test':
            pass

    def __getitem__(self, index):
        key = self.key_list[index]
        img = Image.fromarray(self.image_feature[key][()])
        age = np.array(self.age_dict[key.encode('ascii')])[np.newaxis]
        sex = np.array(self.sex_dict[key.encode('ascii')])[np.newaxis]
        if self.mode.endswith('ex'): # ex binary class of age -> bce
            self.bins = np.array([0, 30, 120])
            age = np.digitize(age, self.bins) - 1
            if age <= -1:
                age = np.zeros_like(age)
            age = torch.FloatTensor(age)
        else: # non-ex 12 class of age -> ce
            self.bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
            age = np.digitize(age, self.bins) - 1
            if age <= -1:
                age = np.zeros_like(age)
            age = torch.tensor(age)
            # age = quantize_age(age, self.bins)[0]
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(sex), age

        # # comparied with Colored MNIST to encode, right now is 1*12 onehot vector
        # bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
        # age = np.digitize(age, bins) - 1
        # if age <= -1:
        #     age = 0

        # if self.transform is not None:
        #     img = self.transform(img)
        
        # return img, torch.FloatTensor(sex), torch.tensor(age).squeeze_()

    def __len__(self):
        return len(self.key_list)

        
def quantize_age(age_vec, bins):
    """
    Get binary vector associated with age value
    """
    n_samples = age_vec.shape[0]
    n_bins = bins.shape[0] - 1

    age_lb = np.zeros((n_samples, n_bins))
    hh = np.digitize(age_vec, bins) - 1

    for i in range(n_samples):
        age_lb[i, hh[i]] = 1

    return age_lb

    
def new_dataset_test_eb(age_dict, sex_dict, key_list, df, mode):
    m_age_list = []
    for age in range(12):
        m = min(df.loc['female', age], df.loc['male', age])
        m_age_list.append(m)
    mm_age_list = [m_age_list, m_age_list.copy()]
    male_young, female_young, male_old, female_old = [],[],[],[]
    for k in key_list:
        kk = k.encode('ascii')
        sex = sex_dict[kk]
        age = age_dict[kk]
        bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
        age = np.digitize(age, bins) - 1
        if mm_age_list[int(sex)][int(age)] > 0:
            if sex == 0 and age < 3:
                female_young.append(k)
            elif sex == 0 and age > 4:
                female_old.append(k)
            elif sex == 1 and age < 3:
                male_young.append(k)
            elif sex == 1 and age > 4:
                male_old.append(k)
            mm_age_list[int(sex)][int(age)] -= 1
    print(len(male_young), len(female_young), len(male_old), len(female_old))
    res = []
    if mode == 'unbiased_test_eb1':
        mi = min(len(female_young), len(male_old))
        l = np.random.permutation(mi).tolist()
        for a in l:
            res.append(female_young[a])
            res.append(male_old[a])
    elif mode == 'unbiased_test_eb2':
        mi = min(len(male_young), len(female_old))
        l = np.random.permutation(mi).tolist()
        for a in l:
            res.append(male_young[a])
            res.append(female_old[a])
    elif mode == 'unbiased_test_balanced':
        mi = min(len(male_young), len(female_young), len(male_old), len(female_old))
        l = np.random.permutation(mi).tolist()
        for a in l:
            res.append(male_young[a])
            res.append(female_young[a])
            res.append(male_old[a])
            res.append(female_old[a])
    return res








def new_dataset(age_dict, sex_dict, key_list, df):
    m_age_list = []
    for age in range(12):
        m = min(df.loc['female', age], df.loc['male', age])
        m_age_list.append(m)
    mm_age_list = [m_age_list, m_age_list.copy()]
    print(mm_age_list)
    male_young, female_young, male_old, female_old = [],[], [], []
    print(len(key_list))
    for k in key_list:
        kk = k.encode('ascii')
        sex = sex_dict[kk]
        age = age_dict[kk]
        bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
        age = np.digitize(age, bins) - 1
        if mm_age_list[int(sex)][int(age)] > 0:
            if sex == 0 and age < 3:
                female_young.append(k)
            elif sex == 0 and age > 4:
                female_old.append(k)
            elif sex == 1 and age < 3:
                male_young.append(k)
            elif sex == 1 and age > 4:
                male_old.append(k)
            mm_age_list[int(sex)][int(age)] -= 1
    mi = min(len(male_young), len(female_young), len(male_old), len(female_old))
    res = []
    l = np.random.permutation(mi).tolist()
    for a in l:
        res.append(male_young[a])
        res.append(female_young[a])
        res.append(male_old[a])
        res.append(female_old[a])
    return res

def IMDB_split_by_sex(age_dict, sex_dict, key_list):
    from collections import defaultdict
    male, female = defaultdict(int), defaultdict(int)
    for k in key_list:
        k = k.encode('ascii')
        age = age_dict[k]
        bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
        age = np.digitize(age, bins) - 1
        if sex_dict[k] == 1:
            male[age] += 1
        elif sex_dict[k] == 0:
            female[age] += 1
    # print('female:', female)
    # print('male:', male)
    import pandas as pd
    df = pd.DataFrame.from_dict([female, male], dtype=int)
    df = df.reindex(sorted(df.columns), axis=1)
    df['Total']= df.sum(axis=1)
    df.loc[len(df.index)]= df.sum(axis=0)
    # df.rename(index={0: 'female', 1: 'male', 2: 'Total'})
    df.index=['female', 'male', 'Total']
    print(df)
    return df