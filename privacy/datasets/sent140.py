from __future__ import print_function
import warnings
import os
import os.path
import json
import csv
import sys
import h5py
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import torch

class Sent140(torch.utils.data.Dataset):

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

        # labels to strings
        self.index2label = {
            0: "negative",
            1: "positive"
        }
        self.model_name = args.model

        # load data and targets
        self.data, self.targets, self.user_ids, self.target2name_map = self.load_meta_data(args, self.data_path, dataset)
        self.modality="nlp"
        self.retrievals = []
        self.prompt_prefix = ""
        self.prompt_choice = args.prompt_choice

    def __getitem__(self, index):
        input, target, uid = self.data[index], int(self.targets[index]), self.user_ids[index]
        fpath= ""

        return input, target, fpath, index, uid

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
            for data, label, uid in zip(training_dataset.data, training_dataset.targets, training_dataset.user_ids):
                label_str = self.index2label[label]
                example = {
                    "input": data,
                    "label": label_str
                }

                # collect by user in aggregate
                if args.prompt_choice == "random_incontext": 
                    training_users2sents[uid].append(example)
                
                # no privacy, combine all user examples
                elif args.prompt_choice == "random_incontext_noprivacy":
                    training_sents.append(example)
                    training_users2counts[uid] += 1
                else:
                    assert 0, print("Unsupported in-context example selection strategy")

            # next assign in context examples to training examples
            test2examples = []
            test2prompts = []
            test2endings = []
            if args.prompt_choice == "random_incontext_noprivacy":
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
                        assert 0, print("Unsupported in-context example selection strategy")
                user_text = [f"""{entry['input']}\nSentiment:{entry['label']}\n\n#####\n\n""" for entry in user_entry]

                # clean
                user_text = " ".join(user_text).replace("<PAD> ", "").replace("<PAD>", "")
                user_text = user_text.replace("<EOS>", "")
                user_text = user_text.replace(" . ", " ")
                user_text = user_text.replace("  ", " ")
                user_text = user_text.replace("\t", " ")

                user_text = f"Is the sentiment positive or negative?\n\n{user_text}"

                test2examples.append(f"""{user_text}{data}\nSentiment:""")
                test2prompts.append(f"""{user_text}""")
                test2endings.append(f"""{data}\nSentiment:""")
            self.data = test2examples
            self.test2prompts = test2prompts
            self.test2endings = test2endings

        else:
            self.test2prompts = [""]*len(self.data)


    def get_prompt(self, input="", incontext={}):
        if "t0" in self.model_name:
            base_prompt = f"Is this text positive or negative? Text: {input}"
        elif "gpt" in self.model_name:
            if "instruction" in self.prompt_choice:
                self.prompt_prefix = "Is the sentiment positive or negative?\n\n"
                base_prompt = f"{self.prompt_prefix}{input}\nSentiment:"
            else:
                prompt_prefix = ""
                self.prompt_prefix = prompt_prefix
                base_prompt = f"""{prompt_prefix}{input}"""
            
        
        # For the experiment with the "public prompt api"; construct the base prompt using this to reproduce
        public_prompt_prefix = f"""
Sentence: This painting is very nice, the artist is talented.
Sentiment: positive

#####

Sentence: I hated this restaurant, the food smelled bad.
Sentiment: negative

#####

Sentence: He was so annoying, his voice was way too loud.
Sentiment: negative

#####

Sentence: This movie was actually pretty funny.
Sentiment: positive

#####

Sentence: """

        return base_prompt


    def load_meta_data(self, args, data_path, split="train"):
        with open(data_path) as f:
            data = json.load(f)

        datas, labels = [], []
        user_ids = []
        target2name_map = {}

        random.seed(args.seed)

        client_subsample = args.client_subsample
        for i, (k, v) in tqdm(enumerate(data['user_data'].items())):
            r = random.random()
            if r > client_subsample:
                continue
            for item, label in zip(v['x'], v['y']):
                text = item[4]
                datas.append(text)
                labels.append(label)
                user_ids.append(i)

            if args and args.dataset_subsize > 0: 
                if i == args.dataset_subsize:
                    break
        
        print(f"Loaded in {len(datas)} points.")

        return datas, labels, user_ids, target2name_map


    def score_gpt(self, example2preds, remove_prefix=False, prompt_prefix="", fp=""):
        count = 0
        correct = 0
        predictions = Counter()
        punct = [",", ".", "?", "!"]
        users2accuracy = defaultdict(dict)
        for i, (key, value) in enumerate(example2preds.items()):
            if remove_prefix:
                rawpred = value['rawpred']
            else:
                rawpred = value['pred']
            label = value['gold']
            input = value['input']

            if remove_prefix:
                # conversion
                if "instruction_prompt" in fp:
                    rawpred = rawpred.replace(self.prompt_prefix, "")
                    rawpred = rawpred.replace(input, "")
                elif "incontext" in fp:
                    prompt_prefix = value['prompt']
                    ending = self.test2endings[i]
                    prompt_idx = rawpred.index(ending)
                    rawpred = rawpred[prompt_idx+len(ending):]
                rawpred = rawpred.replace("Sentiment: ", "")
                rawpred = rawpred.replace("\t", " ")
                rawpred = rawpred.replace("\n", " ")
                rawpred = rawpred.strip()
                rawpred = rawpred.split()
                if rawpred:
                    rawpred = rawpred[0]
                else:
                    rawpred = ""
            
            # cleaning
            if len(rawpred.split()) > 1:
                rawpred = rawpred.split()[0]
            rawpred = rawpred.lower()
            for p in punct:
                rawpred = rawpred.strip(p) 
            rawpred = rawpred.strip(" ") 
            rawpred = rawpred.strip("\t")
            rawpred = rawpred.strip("\n")
            rawpred = rawpred.strip("#")
            
            # mapping
            if rawpred == "positive" or rawpred in ["good", "happy", "1"]:
                pred = 1
            elif rawpred == "negative" or rawpred in ["bad", "sad", "0"]:
                pred = 0
            else:
                pred = rawpred
                predictions[pred] += 1
            
            error = pred == label
            correct += error
            count += 1

            if "uid" in value and value['uid'] not in users2accuracy:
                users2accuracy[value['uid']] = {
                    "correct": 0,
                    "count": 0,
                    "prompt": [],
                    "pred": []
                }
            if "uid" in value:
                users2accuracy[value['uid']]['correct'] += error
                users2accuracy[value['uid']]['count'] += 1
                users2accuracy[value['uid']]['prompt'].append(input)
                users2accuracy[value['uid']]['pred'].append(f"Pred: {pred}, Label: {label}")
            
        acc= correct/count
        if fp:
            print(fp)
        print(f"Accuracy: %.3f (%d correct) for %d examples\n" % (acc, correct, count))
        return acc, users2accuracy


    def compute_accuracy(self, results, dataset, args):
        scores = Counter()
        examples2preds = {}
        num_users = 0
        
        for i, (data, result, uid, prompt) in tqdm(enumerate(zip(dataset, results, self.user_ids, self.test2prompts))):
            label = data[1]
            texts = data[0]
            
            # t0 scoring
            if 1:
                if label == 0 and "negative" in result.lower():
                    pred = 0
                    scores['correct'] += 1
                elif label == 1 and "positive" in result.lower():
                    pred = 1
                    scores['correct'] += 1
                elif label == 0 and "positive" in result.lower():
                    pred = 1
                    scores['incorrect'] += 1
                elif label == 1 and "negative" in result.lower():
                    pred = 0
                    scores['incorrect'] += 1
                else:
                    pred = result
            if i >= len(results):
                break

            examples2preds[i] = {
                "input": texts,
                "pred": pred,
                "gold": label,
                "rawpred": result,
                "prompt": prompt,
                "uid": uid
            }

        # gpt requires removing the prompt prefix based on how it's decoded
        if "gpt" in self.model_name:
            self.score_gpt(examples2preds, remove_prefix=True, fp=args.prompt_choice)

        else:
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
        fpath = f"{results_dir}/{args.paradigm}_{args.model}_{args.split}_sample{args.client_subsample}_{prompt_str}_example2preds.json"
        with open(fpath, "w") as f:
            json.dump(examples2preds, f)

        print(f"Saved results to: {fpath}")
    
