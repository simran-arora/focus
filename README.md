# Can Foundation Models Help Us Achieve Perfect Secrecy

### Code updates coming soon!

This code is for benchmarking FMs of various sizes and types across federated learning tasks.  

1. Use the following commands to clone and install this package. We highly recommend you use conda environments.

```
conda create -n py37 python=3.7
conda activate py37
git clone git@github.com:simran-arora/privacy_fm.git
cd privacy_fm
pip install -e .
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Download benchmark datasets to the ``privacy_fm/benchmarks/'' directory
```
mkdir benchmarks/
cd benchmarks/
```

The LEAF Federated Learning benchmark suite provides: Sent140, Reddit, FEMNIST, and CELEB-A
- Clone this repo in the ``benchmarks/`` directory: https://github.com/TalwalkarLab/leaf
- [Sent140] Go to leaf/data/sent140/ and run ``./preprocess.sh -s niid --sf 1.0 -k 0 -t sample``
- [Reddit] Go to leaf/data/reddit/ and follow the exact download instructions
- [FEMNIST] Go to leaf/data/femnist/ and run ``./preprocess.sh -s niid --sf 1.0 -k 0 -t sample``
- [CELEB-A] Go to leaf/data/celeba/, then download the celebrity faces dataset and attributes files from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html using the resources in the Baidu drive Img and Anno folders as instructed in the leaf repo, then run ``./preprocess.sh -s niid --sf 1.0 -k 0 -t sample``

The FedNLP Federated Learning benchmark suite provides: 20News, MRQA
- Clone this repo in the ``benchmarks/`` directory and go to this path: https://github.com/FedML-AI/FedNLP/tree/27f3f97c72e7f206f8937fe6bcbba39ce79fbcd6/data/download_scripts/text_classification/20Newsgroups
- [20News] run ``bash download_and_unzip.sh`` using their provided script
- [MRQA] Go to https://github.com/FedML-AI/FedNLP/tree/27f3f97c72e7f206f8937fe6bcbba39ce79fbcd6/data/raw_data_loader/MRQA and run ``python download.py`` using their provided script

The FedML Federated Learning benchmark suite provides: CIFAR-10
- Clone this repo in the ``benchmarks/`` directory: https://github.com/FedML-AI/FedML/tree/fedcv
- [CIFAR-10] Go to data/cifar10/ and run ``bash download_cifar10.sh``

3. The ``neurips_privacy/scripts/`` provides scripts to run experiments.

To run CIFAR10 similarity search with CLIP, run:
```
bash scripts/cifar.sh
```

To run Sent140 similarity search with bi-encoder, prompting with T0 (zero shot), and/or in-context learning with GPT3 (zero or few shot), use:
```
bash scripts/sent140.sh
```

To run CelebA similarity search with CLIP, run:
```
bash scripts/celeba.sh
```

To run Reddit prompting with GPT3 (zero or few shot), use:
```
bash scripts/reddit.sh
```

To run 20News similarity search with bi-encoder, prompting with T0 (zero shot), and/or prompting with GPT3 (zero shot), use:
```
bash scripts/news20.sh
```

To run Femnist similarity search with CLIP, run:
```
bash scripts/femnist.sh
```

For MRQA, we use the OpenAI API.

The arguments are described in ``main.py`` and the key decisions are the dataset and model type.
