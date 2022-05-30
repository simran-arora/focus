from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import csv
from torchvision import transforms
from tqdm import tqdm
import json

class CelebA():

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, args, dataset='train', transform=None, target_transform=None, imgview=False):
        
        self.data_file = dataset # 'train', 'test', 'validation'
        self.root = root

        # this is the path to the data downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 
        self.image_path = f"{args.public_dataset_path}/leaf/data/celeba/img_align_celeba"
        
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file)
        self.retrievals = None

        # load data and targets
        self.data, self.targets, self.user_ids = self.load_meta_data(args, self.path)
        self.modality="image"

        self.imgview = imgview

    def __getitem__(self, index):
        imgName, target = self.data[index], int(self.targets[index])
        fpath = os.path.join(self.image_path, imgName)
        img = Image.open(fpath)
        img = self.transform(img)

        return img, target, fpath, index

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, args, path):
        datas, labels = [], []
        user_ids = []
        queries_text = {
            0: "frowning",
            1: "smiling"
        }

        with open(self.root) as f:
            dataset = json.load(f)

        error_images = 0
        for i, ((k, v), username) in tqdm(enumerate(zip(dataset['user_data'].items(), dataset['users']))):
            for j, (x, y) in enumerate(zip(v['x'], v['y'])):
                fpath = f"{self.image_path}/{x}"
                if not os.path.exists(fpath):
                    continue
                elif "144224" in fpath:
                    # image did not exist in download
                    continue
                datas.append(x)
                labels.append(queries_text[y])
                user_ids.append(username)

        index2label = {}
        label2index = {}
        labels_int = []
        for j, lab in enumerate(set(labels)):
            index2label[j] = lab
            label2index[lab] = j
        self.index2label = index2label
        self.label2index = label2index

        print(f"Loaded in {len(datas)} data points.")
        return datas, labels, user_ids

    def load_file(self, path):
        with open(self.root) as f:
            datas = json.load(f)
        return datas
