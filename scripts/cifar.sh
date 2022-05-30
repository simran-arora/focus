python -m privacy.main \
    --dataset cifar10 \
    --model clip32B \
    --paradigm similarity \
    --split test \
    --seed 0 \
    --use_gpu 1 \
    --clip_method zeroshot