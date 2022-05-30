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
import torch
import random
import math

class Reddit():

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

    def __init__(self, data_path, args, dataset='train', transform=None, tokenizer=None):
        
        self.data_file = dataset 
        self.data_path = data_path

        self.index2label = {}

        # load data and targets
        self.data, self.targets, self.user_ids, self.subreddits, self.target2name_map = self.load_meta_data(args, self.data_path, dataset)
        self.modality="nlp"
        self.retrievals = []

    def __getitem__(self, index):
        input, target = self.data[index], self.targets[index]
        fpath= ""
        return input, target, fpath, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_path)))

    def get_incontext_examples(self, args, training_dataset):
        # for in context random
        if "incontext" in args.prompt_choice:

            # first collect incontext examples using the desired strategy
            if args.prompt_choice == "random_incontext":
                training_users2sents = defaultdict(list)
            elif args.prompt_choice == "random_incontext_noprivacy":
                training_sents = []
                training_users2counts = defaultdict(int)
            else:
                training_users2sents = defaultdict(dict)
            for data, label, uid, subreddit in zip(training_dataset.data, training_dataset.targets, training_dataset.user_ids, training_dataset.subreddits):
                example = {
                    "input": data,
                    "label": label
                }

                # collect by user in aggregate
                if args.prompt_choice == "random_incontext": 
                    training_users2sents[uid].append(example)
                
                # no privacy, combine all user examples
                elif args.prompt_choice == "random_incontext_noprivacy":
                    training_sents.append(example)
                    training_users2counts[uid] += 1

                # collect by user, split by subreddit
                else:
                    if subreddit not in training_users2sents[uid]:
                        training_users2sents[uid][subreddit] = []
                    training_users2sents[uid][subreddit].append(example)

            # next assign in context examples to training examples
            test2examples = []
            if args.prompt_choice in ["random_incontext_noprivacy", "incontext"]:
                random.seed(0)
            for data, uid in zip(self.data, self.user_ids):
                # get the incontext examples
                if args.prompt_choice == "random_incontext_noprivacy":
                    user_count = training_users2counts[uid]
                    user_entry = random.sample(training_sents, min(user_count, args.num_incontext))
                else:
                    train_sents = training_users2sents[uid]
                    if args.prompt_choice == "random_incontext":
                        user_entry = train_sents[0:args.num_incontext]
                    else:
                        user_entry = []
                        leftovers = []
                        total_train = sum([len(lst) for sbr, lst in train_sents.items()])
                        for subreddit, lst in train_sents.items():
                            num = math.ceil((len(lst)/total_train)*(args.num_incontext))
                            user_entry.extend(random.sample(lst, min(len(lst), num)))
                        if len(user_entry) > args.num_incontext:
                            user_entry = random.sample(user_entry, args.num_incontext)
                user_text = [f"{entry['input']} {entry['label']}." for entry in user_entry]

                # clean
                user_text = " ".join(user_text).replace("<PAD> ", "").replace("<PAD>", "")
                user_text = user_text.replace("<EOS>", "")
                user_text = user_text.replace(" . ", " ")
                user_text = user_text.replace("  ", " ")
                user_text = user_text.replace("\n", " ")
                user_text = user_text.replace("\t", " ")

                test2examples.append(f"{user_text}{data}")
            self.data = test2examples


    def get_prompt(self, input="", incontext={}):
        prefix = ""
        base_prompt = f"{prefix}{input}"

        return base_prompt


    def clean_text(self, text):
        text = [t for t  in text if t not in ["<BOS>", "<EOS>"]]
        return text


    def load_meta_data(self, args, data_path, split="train"):
        with open(data_path) as f:
            data = json.load(f)

        datas, labels = [], []
        user_ids = []
        target2name_map = {}
        subreddits = []

        for i, (k, v) in tqdm(enumerate(data['user_data'].items())):
            for item, label in zip(v['x'], v['y']):
                for text, lab in zip(item, label['target_tokens']):
                    text = self.clean_text(text)
                    datas.append(" ".join(text))
                    lab = lab[-1]
                    labels.append(lab)
                    user_ids.append(i)
                    subreddits.append(label['subreddit'])
        
        print(f"Loaded in {len(datas)} points.")

        if args and args.dataset_subsize > 0:
            cutoff = args.dataset_subsize
            datas = datas[0:cutoff]
            labels = labels[0:cutoff]
            user_ids = user_ids[0:cutoff]
            subreddits = subreddits[0:cutoff]

        print(f"Using {len(datas)} points.")

        return datas, labels, user_ids, subreddits, target2name_map


    def remove_prefix(self, pred, input):
        ptr = 0
        for ind, c in enumerate(pred):
            if c == " ":
                continue
            elif ptr == len(input):
                break
            elif input[ptr] == c:
                ptr += 1
            else:
                break
        pred = pred[ind:] 
        return pred


    def compute_accuracy(self, results, dataset, args):
        scores = Counter()
        examples2preds = {}
        num_users = 0

        punct = [".", "?", "!", ","]
        for i, (data, result, uid, reddit) in tqdm(enumerate(zip(dataset, results, self.user_ids, self.subreddits))):
            label = data[1]
            texts = data[0]

            examples2preds[i] = {
                "input": texts,
                "pred": result,
                "gold": label,
                "uid": uid,
                "subreddit": reddit
            }

            # score
            pred =  result
            pred = self.remove_prefix(pred, texts.replace(" ", ""))
            
            try:
                pred = pred.split()[0]
            except:
                print(texts, pred, label)
                continue

            # do not score pad examples  
            if label == "<PAD>":
                continue
                
            if label == "<EOS>" and pred in punct[:-1]:
                scores['correct'] += 1
                continue

            if len(pred) > 1 and any(p in pred for p in punct):
                for p in punct:
                    pred = pred.strip(p)
                
            if pred.strip() != label.strip():
                scores['incorrect'] += 1
            else:
                scores['correct'] += 1

        total_samples = scores['incorrect'] + scores['correct']
    
        print(f"Num users {num_users}")
        print(f"Sample count {total_samples}")
        print(f"Accuracy for {total_samples} samples -  {scores['correct']/total_samples}")

        results_dir = f"{args.result_path}/{args.dataset}/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        prompt_str = args.prompt_choice
        if "incontext" in args.prompt_choice:
            prompt_str += str(args.num_incontext)
        with open(f"{results_dir}/{args.paradigm}_{args.model}_{args.split}_{args.client_subsample}_{prompt_str}_example2preds.json", "w") as f:
            json.dump(examples2preds, f)

        print(f"Saved results to: {results_dir}/{args.paradigm}_{args.model}_{args.split}_{args.client_subsample}_{prompt_str}_example2preds.json")

    
