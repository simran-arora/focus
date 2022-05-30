from __future__ import print_function
import warnings
from PIL import Image
import os
import os.path
import csv
from collections import OrderedDict, defaultdict

class FEMNIST():
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
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file)

        # load data and targets
        self.data, self.targets, self.user_ids = self.load_file(args, self.path)
        self.modality = "image"
        self.imgview = imgview

    def __getitem__(self, index):
        imgName, target = self.data[index], int(self.targets[index])
        fpath = os.path.join(self.root, imgName)
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
        datas, labels, user_ids = [], [], []

        client_path = "benchmarks/leaf/data/femnist/client_data_mapping/femnist.csv"
        client2samplepath = defaultdict(list)
        with open(client_path) as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                client = row[0]
                sample_path = row[1]
                client2samplepath[client].append(sample_path)

        samplepath2client = {}
        for cli, samples in client2samplepath.items():
            for sample in samples:
                samplepath2client[sample] = cli

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    label_name = row[-2]
                    bytes_object = bytes.fromhex(label_name)
                    ascii_string = bytes_object.decode("ASCII")

                    x = row[1]
                    y = ascii_string
                    datas.append(x)
                    labels.append(y)
                    user_ids.append(samplepath2client[x])
                line_count += 1

        index2label = {}
        label2index = {}
        labels_int = []

        organize_labels = {}
        for lab in set(labels):
            if lab.isdigit():
                lab_str = f"The picture is of the digit {lab}"
            elif lab.isupper():
                lab_str = f"The picture is of the uppercase letter {lab}"
            else:
                lab_str = f"The picture is of the lowercase letter {lab}"
            if lab_str not in organize_labels:
                organize_labels[lab_str] = lab

        organize_labels_keys = list(organize_labels.keys())
        organize_labels_keys = sorted(organize_labels_keys)
        for j, key in enumerate(organize_labels_keys):
            index2label[j] = organize_labels[key]
            label2index[organize_labels[key]] = j

        self.index2label = index2label
        self.label2index = label2index
        
        for lab in labels:
            labels_int.append(label2index[lab])
            
        return datas, labels_int, user_ids

    def load_file(self, args, path):
        # load meta file to get labels
        datas, labels, user_ids = self.load_meta_data(args, os.path.join(self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))
        return datas, labels, user_ids
