# Sent140
cd leaf/data/sent140
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample

# FEMNIST
cd ../femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample

# CELEB-A
cd ../celeba
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample

# 20News
cd ../../../
rm -rf 20news-bydate-test
rm -rf 20news-bydate-train
rm 20news-bydate.tar.gz
wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -zxvf 20news-bydate.tar.gz
rm 20news-bydate.tar.gz

# CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
