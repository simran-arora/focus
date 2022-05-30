import json
import os
import csv
import sys
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from sklearn.linear_model import LogisticRegression

import torch
from privacy.clip import clip
from PIL import Image
import glob


def classify_embeddings(clip_model,
                        image_embeddings, 
                        text_descriptions,
                        args,
                        temperature=100.):
    
    text_tokens = clip.tokenize(text_descriptions)
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        _image_embeddings = (image_embeddings / 
                             image_embeddings.norm(dim=1, keepdim=True))
        
        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        _text_embeddings = (text_embeddings / 
                            text_embeddings.norm(dim=1, keepdim=True))
        
        cross = temperature * _image_embeddings @ _text_embeddings.T
        text_probs = cross.softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)
    clip_model.cpu()
    return predicted.cpu().numpy()


def logreg(train_embeddings, train_sentences, test_embeddings):
    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    train_labels = [torch.Tensor([entry['gold']]) for entry in train_sentences]
    train_labels = torch.cat(train_labels).cpu().numpy()
    classifier.fit(train_embeddings, train_labels)
    
    predictions = classifier.predict(test_embeddings)
    return predictions


def save_logreg_results(args, sentences, predictions, queries_text):
    test_labels = [torch.Tensor([entry['gold']]) for entry in sentences]
    test_labels = torch.cat(test_labels).cpu().numpy()
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

    correct = defaultdict(list)
    example2pred = {}
    for i,(sentence, pred) in enumerate(zip(sentences, predictions)):
        gold = sentence['gold']
        gold = int(gold)
        pred = int(pred)
        correct[gold].append(gold == pred)
        example2pred[i] = {
            "pred": queries_text[pred],
            "gold": queries_text[gold],
            "input": sentence["imgpath"]
        }

    total_crct = []
    cls2acc = {}
    for key, value in correct.items():
        label_name = queries_text[key]
        total_crct.extend(value)
        acc = len([c for c in value if c == True])/len(value)
        cls2acc[label_name] = acc
        print(f"Label: {label_name}, Accuracy: {acc}, for {len(value)} examples.")
    print()
    total_acc = len([c for c in total_crct if c == True])/len(total_crct)
    cls2acc['total'] = total_acc
    print(f"Total: {total_acc}, for {len(total_crct)} examples.")

    if not os.path.exists(f"results_prompting/{args.dataset}/"):
        os.makedirs(f"results_prompting/{args.dataset}/")
    with open(f"results_prompting/{args.dataset}/{args.clip_method}_{args.model}_example2preds.json", "w") as f:
        json.dump(example2pred, f)
    with open(f"results_prompting/{args.dataset}/{args.clip_method}_{args.model}_cls2acc.json", "w") as f:
        json.dump(cls2acc, f)


def get_dataset_embeddings(args, dataloader, model, split="test"):
    # File paths
    embeddings_dir = f"{args.embeddings_dir}/{args.dataset}/"
    embedding_fname = f'd={args.dataset}-s={split}-m={args.model}-seed={args.seed}.pt'
    embedding_path = os.path.join(embeddings_dir, embedding_fname)

    # Load existing embeddings
    if os.path.exists(embedding_path):
        print(f'-> Retrieving image embeddings from {embedding_path}!')
        embeddings = torch.load(embedding_path)
        sentences = prepare_data(dataloader)
        return torch.load(embedding_path), sentences

    # Compute the dataset embeddings now
    all_embeddings = []
    model.to(args.device)
    model.eval()
    count = 0
    with torch.no_grad():
        sentences = []
        for ix, data in enumerate(tqdm(dataloader, 
                                       desc=f'Computing CLIP image embeddings for {split} split')):

            if args.dataset != "cifar10":
                inputs, labels, fpath, data_ix = data
            else:
                inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            embeddings = model.encode_image(inputs).float().cpu()
            all_embeddings.append(embeddings)
            inputs = inputs.cpu()
            labels = labels.cpu()

            for i in range(inputs.shape[0]):
                # label_name = dataset.target2name_map[label]
                if args.dataset != "cifar10":
                    entry = { "exidx": int(data_ix[i]), "gold": int(labels[i]), "imgpath": fpath[i]}
                else:
                    entry = { "exidx": count, "gold": int(labels[i]), "imgpath": ""}
                    count += 1
                sentences.append(entry)
    model.cpu()
    
    # Save to disk
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    torch.save(torch.cat(all_embeddings), embedding_path)
    print(f'-> Saved embeddings to {embedding_path}!')
    return torch.cat(all_embeddings), sentences


def prepare_data(dataloader):
    sentences = []
    count = 0
    for bix, data in tqdm(enumerate(dataloader)):
        for i in range(len(data[0])):
            input = data[0][i]
            label = data[1][i]
            if len(data) > 2:
                fpath = data[2][i]
                exidx = data[3][i]
            else:
                exidx = count
                count += 1
                fpath = ""
            entry = { "exidx": exidx, "gold": label, "imgpath":fpath}
            sentences.append(entry)
    return sentences


def save_results(args, sentences, predictions, queries_text):
    correct = defaultdict(list)
    example2pred = {}
    for i,(sentence, pred) in enumerate(zip(sentences, predictions)):
        gold = sentence['gold']
        gold = int(gold)
        correct[gold].append(gold == pred)

        example2pred[i] = {
            "pred": queries_text[pred],
            "gold": queries_text[gold],
            "input": sentence["imgpath"]
        }

    total_crct = []
    cls2acc = {}
    for key, value in correct.items():
        label_name = queries_text[key]
        total_crct.extend(value)
        acc = len([c for c in value if c == True])/len(value)
        cls2acc[label_name] = acc
        print(f"Label: {label_name}, Accuracy: {acc}, for {len(value)} examples.")
    print()
    total_acc = len([c for c in total_crct if c == True])/len(total_crct)
    cls2acc['total'] = total_acc
    print(f"Total: {total_acc}, for {len(total_crct)} examples.")

    if not os.path.exists(f"results_prompting/{args.dataset}/"):
        os.makedirs(f"results_prompting/{args.dataset}/")
    with open(f"results_prompting/{args.dataset}/{args.clip_method}_example2preds.json", "w") as f:
        json.dump(example2pred, f)
    with open(f"results_prompting/{args.dataset}/{args.clip_method}_cls2acc.json", "w") as f:
        json.dump(cls2acc, f)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def run_image_similarity_classification(args, model, dataset, train_dataset):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, timeout=60, num_workers=1, drop_last=False)
    dataset_embeddings, sentences = get_dataset_embeddings(args, dataloader, model, split=dataset.split)

    if args.clip_method == "zeroshot":  
        queries_text = dataset.index2label
        candidate_captions = [v for k, v in queries_text.items()]
        predictions = classify_embeddings(model, dataset_embeddings, candidate_captions, args) 
        save_results(args, sentences, predictions, queries_text)   

    elif args.clip_method == "logreg":
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, timeout=60, num_workers=1, drop_last=False)
        train_dataset_embeddings, train_sentences = get_dataset_embeddings(args, train_dataloader, model, split=train_dataset.split)
        predictions = logreg(train_dataset_embeddings, train_sentences, dataset_embeddings) 
        queries_text = dataset.index2label
        save_logreg_results(args, sentences, predictions, queries_text)
        

      

    
    