python -m privacy.main \
    --dataset celeba \
    --model clip32B \
    --paradigm similarity \
    --split test  \
    --use_gpu 1 \
    --seed 0 \
    --clip_method zeroshot