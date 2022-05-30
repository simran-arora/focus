 python -m privacy.main \
    --dataset femnist \
    --model clip32B \
    --paradigm similarity \
    --split test  \
    --use_gpu 1  \
    --clip_method zeroshot \
    --seed 0