# Similarity search (zero shot) with bi-encoder model
python -m privacy.main \
    --dataset 20news  \
    --model dpr  \
    --paradigm similarity \
    --split test \
    --use_gpu 1 \
    --seed 0 


# Run zero-shot with GPT
for MODEL in gpt125m gpt1.3 gpt2.7;  
do
    for NUM in 0; 
    do
        BATCH_SIZE=4
        if [[ $MODEL == "gpt1.3" ]]; then
            BATCH_SIZE=16
        elif [[ $MODEL == "gpt2.7" ]]; then 
            BATCH_SIZE=8
        elif [[ $MODEL == "gpt125m" ]]; then 
            BATCH_SIZE=32
        fi
        
        echo "Model ${MODEL}" 
        echo "Batch Size ${BATCH_SIZE}" 
        echo "Num in context ${NUM}" 
        python -m privacy.main \
            --dataset 20news \
            --model ${MODEL} \
            --paradigm prompt \
            --split test \
            --batch_size ${BATCH_SIZE} \
            --use_gpu 1 \
            --seed 0 \
            --max_sequence_length 1800 
    done
done


# Command for GPT OpenAI API Inference 
python -m privacy.main \
    --dataset 20news \
    --model gpt6.7 \
    --paradigm prompt \
    --split test \
    --batch_size 1 \
    --seed 0 \
    --openai_key "fill in"