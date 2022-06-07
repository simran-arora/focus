import torch
import sys
import os
import logging
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
import torchvision
from collections import defaultdict
from tqdm import tqdm
from torchvision import datasets, transforms

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification
from transformers import  GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM

import torch
from privacy.clip import clip
from PIL import Image
import glob

from privacy.datasets.celeba import CelebA
from privacy.datasets.sent140 import Sent140
from privacy.datasets.news20 import News20
from privacy.datasets.femnist import FEMNIST
from privacy.datasets.reddit import Reddit

API_MODELS = ["gpt175", "gpt6.7"]


def get_model(args):
    print("Loading model...")
    if "clip" in args.model:
        if args.model == "clip32B":
            clip_variant = "ViTB32"
        elif args.model == "clip16B":
            clip_variant = "ViTB16"
        elif args.model == "clip336":
            clip_variant = "ViTL14"
        elif args.model == "clipres101":
            clip_variant = "RN101"
        else:
            assert 0, print("Unsupported clip variant")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, transform = clip.load(clip_variant, device=device)
        tokenizer = None
    elif args.model == "dpr":
        tokenizer = None
        transform = None
        model = SentenceTransformer(("multi-qa-mpnet-base-dot-v1"))
    elif "t0" in args.model:
        # T0 3Bn Model
        transform = None 
        if args.model == "t0pp":
            t0_variant = "bigscience/T0pp"
        elif args.model == "t03b":
            t0_variant = "bigscience/T0_3B"
        else:
            assert 0, print("Unsupported t0 variant.")
        tokenizer = AutoTokenizer.from_pretrained(t0_variant, cache_dir=args.cache_dir)
        tokenizer.padding_side = "left"
        model = AutoModelForSeq2SeqLM.from_pretrained(t0_variant, cache_dir=args.cache_dir)
    elif "gpt" in args.model:
        transform = None

        if args.model in API_MODELS:
            return None, None, None 

        if args.model == "gpt2.7":
             gpt_variant = 'EleutherAI/gpt-neo-2.7B'
        elif args.model == "gpt1.3":
            gpt_variant = 'EleutherAI/gpt-neo-1.3B'
        elif args.model == "gpt125m":
            gpt_variant = 'EleutherAI/gpt-neo-125M'
        else:
            assert 0, print("Unsupported gpt variant.")

        tokenizer = AutoTokenizer.from_pretrained(gpt_variant, max_token_length=512, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            gpt_variant, 
            pad_token_id=tokenizer.eos_token_id, 
            cache_dir=args.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif "bert" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        transform = None
        model = None
    else:
        assert 0, print("Unsupported model.")

    if args.use_gpu:
        model = model.to(args.device)

    return model, transform, tokenizer


def get_dataset(args, split="", transform=None, tokenizer=None):
    print("\nLoading dataset...")
    dataset = args.dataset

    if dataset == "sent140":
        data_prefix = f"{args.public_datasets_prefix}/leaf/data/"
        data_path = f"{data_prefix}/sent140/data/train/all_data_niid_0_keep_0_train_9.json"
        training_dataset = Sent140(data_path, args, dataset="train")

        data_path = f"{data_prefix}/sent140/data/test/all_data_niid_0_keep_0_test_9.json"
        test_dataset = Sent140(data_path, args, dataset="test")
        test_dataset.get_incontext_examples(args, training_dataset)

        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform

    elif dataset == "reddit":
        data_prefix = f"{args.public_datasets_prefix}/leaf/data/"
        data_path = f"{data_prefix}/reddit/data/train/train_data.json"
        training_dataset = Reddit(data_path, args, dataset="train")

        data_path = f"{data_prefix}/reddit/data/test/test_data.json"
        test_dataset = Reddit(data_path, args, dataset="test")
        test_dataset.get_incontext_examples(args, training_dataset)

        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform

    elif dataset == "celeba":
        data_prefix = f"{args.public_datasets_prefix}/leaf/data/"
        data_path = f"{data_prefix}/celeba/data/train/all_data_niid_0_keep_0_train_9.json"
        training_dataset = CelebA(f'{data_path}', args, dataset="train") 

        data_path = f"{data_prefix}/celeba/data/test/all_data_niid_0_keep_0_test_9.json"
        test_dataset = CelebA(f'{data_path}', args, dataset="test") 
        
        if not transform:
            transform = torchvision.transforms.ToTensor()
        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform

    elif dataset == "femnist":
        data_prefix = f"{args.public_datasets_prefix}/leaf/data/"
        train_transform = transform
        test_transform = transform
        data_path = f"{data_prefix}/femnist/"
        training_dataset = FEMNIST(f'{data_path}', args, dataset="train", transform=train_transform) 
        test_dataset = FEMNIST(f'{data_path}', args, dataset="test", transform=test_transform) 

        training_dataset.tokenizer = tokenizer
        training_dataset.transform = train_transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = test_transform

    elif dataset == "20news":
        data_path = f'{args.public_datasets_prefix}/fedNLP/data_files/20news_data.h5'
        partition_path = f'{args.public_datasets_prefix}/fedNLP/partition_files/20news_partition.h5'
        training_dataset = News20(f'{data_path}', f'{partition_path}', args, dataset="train") 
        test_dataset = News20(f'{data_path}', f'{partition_path}', args, dataset="test") 
        
        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform
        training_dataset.max_length = 1900
        test_dataset.max_length = 1900

    elif dataset == "mrqa":
        data_path = f'{args.public_datasets_prefix}/fedNLP/data_files/mrqa_data.h5'
        partition_path = f'{args.public_datasets_prefix}/fedNLP/partition_files/mrqa_partition.h5'
        training_dataset = News20(f'{data_path}', f'{partition_path}', args, dataset="train") 
        test_dataset = News20(f'{data_path}', f'{partition_path}', args, dataset="test") 
        
        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform
        training_dataset.max_length = 1900
        test_dataset.max_length = 1900

    elif hasattr(torchvision.datasets, dataset.upper()):
        data_path = f"{args.public_datasets_prefix}"
        dataset_name = dataset.upper()
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"] and "clip" not in args.model:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"] and "clip" not in args.model:
            transform = torchvision.transforms.ToTensor()
        
        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=f"{data_path}/cifar10/",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=f"{data_path}/cifar10/",
            train=False,
            download=True,
            transform=transform
        )

        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        num_categories = np.unique(training_dataset.targets).shape[0]
        
        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()

        training_dataset.tokenizer = tokenizer
        training_dataset.transform = transform
        test_dataset.tokenizer = tokenizer
        test_dataset.transform = transform
        training_dataset.modality = "image"
        test_dataset.modality = "image"

        index2label = {
            0: "airplane", 1:"automobile", 2: "bird", 3: "cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"
        }
        label2index = {}
        for j, lab in enumerate(index2label.items()):
            label2index[lab] = j
        training_dataset.index2label = index2label
        test_dataset.index2label = index2label
        training_dataset.label2index = label2index
        test_dataset.label2index = label2index
    
    else:
        assert 0, print("Unsupported dataset.")

    training_dataset.split = "train"
    test_dataset.split = "test"

    return training_dataset, test_dataset, training_dataset.transform


"""
    Load the model
    Load the dataset
"""
def initialize_run(args):
    transform = None
    tokenizer = None
    
    # get dataset and model
    model, transform, tokenizer = get_model(args)
    print(f"Loaded model: {args.model}")
    if model:
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters()) 
        print("Total params ", total_params, " total trainable params ", total_trainable_params)
    training_dataset, test_dataset, transform = get_dataset(args, transform=transform, tokenizer=tokenizer)  
    num_categories = np.unique(training_dataset.targets).shape[0]   

    return training_dataset, test_dataset, model


def index_dataset(args, index_split="train", search_split="test"):
    train_data = get_dataset(args, split=index_split)
    index_sents = []
    for i, data in enumerate(train_data):
        entry = {
            "text": data[0],
            "label_no": data[1],
            "ex_idx": data[-1]
        }
        index_sents.append(entry)
    eval_data = get_dataset(args, split=search_split)
    eval_sents = []
    for i, data in enumerate(eval_data):
        entry = {
            "text": data[0],
            "label_no": data[1],
            "ex_idx": data[-1]
        }
        eval_sents.append(entry)

    retrieval_model = SentenceTransformer(("multi-qa-mpnet-base-dot-v1"))

    print("Indexing ...")
    corpus, index = index_trainset(index_sents, retrieval_model, args)

    print("Searching ...")
    results = search_trainset(eval_sents, index, corpus, top_k=1)
    preds, results = compute_preds(results, corpus)

    return preds, results  


def get_zeroshot_predictions(key_embeddings, 
                             query_embeddings,
                             temperature=100.,
                             numpy=True,
                             base_model=None,
                             normalize_query=False):
    
    key_embeddings = torch.from_numpy(key_embeddings)
    query_embeddings = torch.from_numpy(query_embeddings)

    with torch.no_grad():
        _key_embeddings = (key_embeddings / 
                           key_embeddings.norm(dim=-1, keepdim=True))
        if normalize_query is True:
            _query_embeddings = (query_embeddings / 
                                 query_embeddings.norm(dim=-1, keepdim=True))
        else:
            _query_embeddings = query_embeddings
        
        cross = _key_embeddings @ _query_embeddings.T
        probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(probs.data, 1)

    return predicted.cpu().numpy()

