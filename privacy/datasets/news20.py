from __future__ import print_function
import warnings
import os
import os.path
import json
import csv
import sys
import h5py
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import random
import torch


class News20():

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

    def __init__(self, data_path, partition_path, args, dataset='train'):
        
        self.data_file = dataset 
        self.data_path = data_path
        self.partition_path = partition_path

        # load data and targets
        self.data, self.targets, self.user_ids = self.load_meta_data(args, self.data_path, self.partition_path, dataset)

        if args:
            self.model_name = args.model
        self.retrievals = None
        self.modality = "nlp"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target) where target is index of the target class.
        """

        input, target = self.data[index], self.targets[index]
        fpath= ""
        uid = self.user_ids[index]

        return input, target, fpath, index, uid

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_path)))


    def get_prompt(self, input="", label="", incontext={}):
        if "gpt" in self.model_name:
            selected_topics = [value for key, value in self.index2label.items()]
            instruction = "Is the topic "
            for topic in selected_topics[:-1]:
                instruction += f"{topic}, "
            instruction += f"or {selected_topics[-1]}?\n\n"

            clean_input = self.parse_document(input)['lines']
            clean_input = self.clean_lines(clean_input)

            token_text = self.tokenizer(clean_input, return_tensors="pt", truncation=True, padding=True)
            token_text = self.tokenizer.convert_ids_to_tokens(token_text.input_ids[0])
            token_text = token_text[:self.max_length]
            token_text = " ".join(token_text)

            truncated_text = clean_input[:len(token_text)]
            clean_input = f"{instruction}{truncated_text}"
            self.instruction_end = "\nTopic:"
            base_prompt = f"{clean_input}{self.instruction_end}"

        elif "t0" in self.model_name:
            selected_topics = [value for key, value in self.index2label.items()]
            instruction = "What label best describes text ("
            for topic in selected_topics[:-1]:
                instruction += f"{topic} ||| "
            instruction += f"||| {selected_topics[-1]})?"

            clean_input = self.parse_document(input)['lines']
            clean_input = self.clean_lines(clean_input)
            token_text = self.tokenizer(clean_input, return_tensors="pt", truncation=True, padding=True)
            token_text = self.tokenizer.convert_ids_to_tokens(token_text.input_ids[0])
            token_text = token_text[:self.max_length]
            token_text = " ".join(token_text)
            truncated_text = clean_input[:len(token_text)]

            base_prompt = f"Text: {truncated_text} \n{instruction}\n"
        else:
            assert 0, print("Unsupported operation in 20news.")

        return base_prompt


    def parse_document(self, doc):
        key_parts = ["Subject: Re:", "Organization", "Lines"]
        parts_dict = {}
        
        parts = doc.split(key_parts[0])
        parts_dict['from'] = parts[0]
        
        try:
            parts = parts[1].split(key_parts[1])
        except:
            parts = doc.split(key_parts[1])
        parts_dict['subject'] = parts[0]
        
        try:
            parts = parts[1].split(key_parts[2])
        except:
            parts = doc.split(key_parts[2])
        parts_dict['org'] = parts[0]
        
        try:
            parts_dict['lines'] = parts[1][4:]
        except:
            parts = doc.split(key_parts[2])
            try:
                parts_dict['lines'] = parts[1][4:]
            except:
                parts_dict['lines'] = doc
        
        return parts_dict


    def clean_lines(self, lines):
        lines = lines.replace("*", "")
        lines = lines.replace("|", "")
        lines = lines.replace(">", "")
        lines = lines.replace("<", "")
        lines = lines.replace("---", "")
        lines = lines.replace("^", "")
        lines = lines.replace("\t", "")
        
        clean_lines = []
        for wd in lines.split():
            if "@" not in wd and ".com" not in wd:
                clean_lines.append(wd)
                
        lines = " ".join(clean_lines)
        
        ines = lines.replace("   ", " ")
        lines = lines.replace("  ", " ")
        return lines    

    # convert raw label names to label descriptions
    def clean_label(self, label):
        category2word = {
                "comp": "",
                "alt": "",
                "misc": "",
                "sci": "",
                "talk": "",
                "rec": "",
                "soc": "",
                "sport": "",
                "autos": "automobiles",
                "med": "medical",
                "crypt": "cryptography security",
                "mideast": "middle east",
                "sys": "",
                "forsale": "sale",
        }
                
        label_clean = label.replace("religion.christian", "christianity")
        wds = label_clean.split(".")
        clean_wds = []
        for wd in wds:
            if wd in category2word:
                wd = category2word[wd]
                if wd:
                    clean_wds.append(wd)
            else:
                clean_wds.append(wd)
        label_text =  " ".join(clean_wds)
        return label_text


    def get_news20_iserror(self, topics, result, gold):
        error = 0
        result = result.split(".")[0].strip()
        result = result.lower()
        found_topics = [topic for topic in topics if topic in result]
        if not result or gold.lower() not in result.split():
            error = 1
        elif len(found_topics) > 1:
            error = 1
        
        if len(found_topics) > 1:
            if "politics" in found_topics and result != "politics" and gold != "politics" and "politics" in gold and gold in result:
                error = 0    
        result_fix = result + "s" 
        if result_fix == gold:
            error = 0
        elif result == "medicine" and gold == "medical":
            error = 0
        elif result == "cars" and gold == "automobiles":
            error = 0

        return error, result


    def compute_accuracy(self, results, dataset, args):
        scores = Counter()
        examples2preds = {}
        num_users = 0
        errors = 0
        count = 0
        topics = [topic for index, topic in self.index2label.items()]

        for i, (data, result) in enumerate(zip(dataset, results)):
            gold = data[1]
            gold = self.index2label[gold]
            texts = data[0]
            error = 0
        
            if "gpt" in args.model:
                # need to chop off the prompt
                prompt_idx = result.index(self.instruction_end)
                result = result[prompt_idx+len(self.instruction_ind):]

            error, result = self.get_news20_iserror(topics, result, gold)
            results.append(result)
            errors += error
            count += 1

            examples2preds[i] = {
                "text": texts,
                "label": gold,
                "pred": result
            }

            # for inspection
            if not error and i < 100:
                print(result, gold)
                print("---------------------------------")
            
        print(f"Accuracy: {1 - (errors/count)} across {count} examples.")

        directory = f"{args.result_path}/{args.dataset}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{args.paradigm}_{args.model}_{args.split}_{args.client_subsample}_examples2preds.json", "w") as f:
            json.dump(examples2preds, f)


    def load_meta_data(self, args, data_path, partition_path, split="train"):
        def read_instance_from_h5(data_file, index_list, desc=""):
            X = list()
            y = list()
            for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
                xval = data_file["X"][str(idx)][()]
                if type(xval) == str:
                    X.append(xval)
                else:
                    X.append(xval.decode("utf-8"))

                yval = data_file["Y"][str(idx)][()]
                if type(yval) == str:
                    y.append(yval)
                else:
                    y.append(yval.decode("utf-8"))
            return {"X": X, "y": y}

        def load_attributes(data_path):
            data_file = h5py.File(data_path, "r", swmr=True)
            attributes = json.loads(data_file["attributes"][()])
            data_file.close()
            return attributes

        attributes = load_attributes(data_path)
        label_vocab=attributes["label_vocab"]
        data_file = h5py.File(data_path, "r", swmr=True)
        attributes = json.loads(data_file["attributes"][()])
        data_file.close()

        data_file = h5py.File(data_path, "r", swmr=True)
        partition_file = h5py.File(partition_path, "r", swmr=True)
        partition_method = "uniform"
        index_list = []
        examples_per_client = []

        split_to_load = split.replace("_split", "")
        if split_to_load == "val":
            split_to_load = "train"
        for client_idx in tqdm(
            partition_file[partition_method]
            ["partition_data"].keys(),
            desc="Loading index from h5 file."):
            index_list.extend(partition_file[partition_method]["partition_data"][client_idx][split_to_load][()][:])
            examples_per_client.append(len(partition_file[partition_method]["partition_data"]
                [client_idx][split_to_load][()][:]))
        data = read_instance_from_h5(data_file, index_list)
        data_file.close()
        partition_file.close()

        print(f"Average examples per client: {sum(examples_per_client)/len(examples_per_client)}")

        # Existing
        datas, labels, user_ids = [], [], []

        random.seed(args.seed)

        if not args:
            client_subsample = 1
        else:
            client_subsample = args.client_subsample

        user_id_count = 0
        user_examples_left = examples_per_client[user_id_count]
        for j, (x, y) in enumerate(zip(data['X'], data['y'])):
            if user_id_count < len(examples_per_client)/2 and split=="val_split":
                continue
            elif user_id_count > len(examples_per_client)/2 and split=="train_split":
                continue
            r = random.random()
            if r > client_subsample:
                continue
                
            datas.append(x)
            labels.append(self.clean_label(y))

            if args and args.dataset_subsize > 0:
                if j == args.dataset_subsize:
                    break

            user_ids.append(user_id_count)

            user_examples_left -= 1
            if user_examples_left == 0:
                user_id_count += 1
                if user_id_count < len(examples_per_client):
                    user_examples_left = examples_per_client[user_id_count]

        index2label = {}
        label2index = {}
        labels_int = []
        for j, lab in enumerate(set(labels)):
            index2label[j] = lab
            label2index[lab] = j

        self.index2label = index2label
        self.label2index = label2index
        
        for lab in labels:
            labels_int.append(label2index[lab])
            
        return datas, labels_int, user_ids
