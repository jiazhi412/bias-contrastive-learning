import os
import re
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. running
def set_random_seed(seed_number):
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)


def str_list(s):
    if type(s) is type([]):
        return s
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [x for x in vals]


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_pkl(pkl_data, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(pkl_data, f)


def load_pkl(load_path):
    with open(load_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def save_dict(dict, save_path, file_name):
    with open((save_path + file_name), "w") as f:
        for key, value in dict.items():
            f.write("   " + str(key) + ": " + str(value) + "\n")
    f.close()

# TODO
def load_dict(save_path, file_name):
    # importing the module
  
    # reading the data from the file
    with open('dictionary.txt') as f:
        data = f.read()
  
    print("Data type before reconstruction : ", type(data))
      
    # reconstructing the data as a dictionary
    js = json.loads(data)
  
    print("Data type after reconstruction : ", type(js))
    print(js)
    

    with open((save_path + file_name), "w") as f:
        for key, value in dict.items():
            f.write("   " + str(key) + ": " + str(value) + "\n")
    f.close()
    return dict

def save_list(list, save_path, file_name):
    with open((save_path + file_name), "w") as f:
        for item in list:
            f.write("%s\n" % item)
    f.close()


def save_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(load_path):
    with open(load_path, "r") as f:
        json_data = json.load(f)
    return json_data


def save_state_dict(state_dict, save_path):
    torch.save(state_dict, save_path)


def write_dict(f, dict):
    for k, v in dict.items():
        f.write("   " + str(k) + ": " + str(v) + "\n")


def write_report(save_path, **kwargs):
    with open(save_path, "w") as f:
        f.write("Performance evaluation: \n")
        write_dict(f, kwargs["performance_info"])
        f.write("\nFairness evaluation: \n")
        for i in range(len(kwargs["fairness_info"])):
            write_dict(f, kwargs["fairness_info"][i])
            f.write("\n")
        f.write("\nParameters: \n")
        write_dict(f, kwargs["opt"])
        print("Save success!")
    f.close()


def write_by_line(save_path, list):
    with open(save_path, "w") as f:
        for item in list:
            f.write("%s\n" % item)
    f.close()
