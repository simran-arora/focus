from setuptools import setup

_REQUIRED = [
    "h5py",
    "transformers==4.17.0",
    "sentence-transformers==2.2.0",
    "matplotlib",
    "ftfy",
    "faiss-gpu==1.7.2",
    "tensorboard==2.8.0"
]

setup(
    name="privacy",
    version="0.0.1",
    description="Code and resources for privacy with foundation models",
    author="Simran Arora",
    author_email="simarora@stanford.edu",
    packages=["privacy"],
    install_requires=_REQUIRED,
)
