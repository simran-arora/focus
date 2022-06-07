# Can Foundation Models Help Us Achieve Perfect Secrecy?

This code is for benchmarking FMs of various sizes and types across federated learning tasks. The paper can be found here: https://arxiv.org/abs/2205.13722  

## Setup

Use the following commands to clone and install this package. We highly recommend you use conda environments.

```
# environment
conda create -n py37 python=3.7
conda activate py37

# installations
git clone git@github.com:simran-arora/focus.git
cd focus
pip install -e .
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

cd benchmarks/leaf
git submodule init
git submodule update
```

## Obtain the datasets

Download benchmark datasets to the ``focus/benchmarks/'' directory
```
cd benchmarks/
bash download_data.sh
```

The LEAF Federated Learning benchmark suite provides: Sent140, Reddit, FEMNIST, and CELEB-A. The FedNLP suite provides 20News and MRQA. The FedML suite provides CIFAR-10.
- Sent140, FEMNIST, CelebA, CIFAR-10, and 20News benchmarks are downloaded via the provided download script. 
- [Reddit] Go to benchmarks/leaf/data/reddit/ and follow the download instructions.
- [MRQA] Go to https://github.com/FedML-AI/FedNLP/tree/27f3f97c72e7f206f8937fe6bcbba39ce79fbcd6/data/raw_data_loader/MRQA and run ``python download.py`` using their provided script.

## Run the code

The ``focus/scripts/`` directory provides scripts to run experiments for each benchmark.

For example:
```
bash scripts/cifar.sh
bash scripts/sent140.sh
```


## Citation
Please use the following Bibtex for this work:
```
@misc{arora2022focus,
      title={Can Foundation Models Help Us Achieve Perfect Secrecy?}, 
      author={Simran Arora and Christopher Ré},
      year={2022},
      eprint={2205.13722},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
