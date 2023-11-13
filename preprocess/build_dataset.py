import os
import random
import shutil
import yaml


data_config_file = "../data/config.yaml"
with open(data_config_file, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
path = config['raw_dataset']
dataset_path =  config['data_root']
labels = os.listdir(path=path)
nums_per_label = config['num_per_class']
ratio = config['ratio']
print(labels)
types = ["train", "test"]
for t in types:
    if not os.path.exists(dataset_path+t):
        os.makedirs(dataset_path+t)
nums = 0
for label in labels:
    for t in types:
        files = os.listdir(path=path+label)
        random.shuffle(files)
        data_path = dataset_path+t+"/"+label
        if not os.path.exists(dataset_path+t+"/"+label):
            os.makedirs(dataset_path+t+"/"+label)
        # train datasets
        if t == "train":
            for file_name in files[:int(nums_per_label*ratio)]:
                shutil.copy(path+"/"+label+"/"+file_name, data_path)
                nums += 1
        # test datasets
        else:
            for file_name in files[int(nums_per_label*ratio):nums_per_label]:
                shutil.copy(path+"/"+label+"/"+file_name, data_path)
                nums += 1
print("total: ", nums)