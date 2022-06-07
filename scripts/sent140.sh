# Command for T0 (t03b, t0pp -- we ran the latter on CPUs, another option you can use is the HuggingfaceAPI https://huggingface.co/inference-api)
python -m privacy.main \
    --dataset sent140 \
    --model t03b \
    --paradigm prompt \
    --split test \
    --batch_size 4 \
    --use_gpu 1 \
    --seed 0 \
    --client_subsample 0.025


# Command for Bi-Encoder
python -m privacy.main \
    --dataset sent140 \
    --model dpr \
    --paradigm similarity \
    --use_gpu 1 \
    --seed 0 \
    --client_subsample 0.025
    --split test


# Command for GPT Variations with "User Privacy" In context examples
for MODEL in gpt2.7 gpt1.3 gpt125m; 
do
    for NUM in 3 5; 
    do
        if [[ $MODEL == "gpt1.3" ]]; then
            BATCH_SIZE=32
        elif [[ $MODEL == "gpt125m" ]]; then 
            BATCH_SIZE=72
        elif [[ $MODEL == "gpt2.7" ]]; then 
            BATCH_SIZE=24
        fi
        
        echo "Model ${MODEL}" 
        echo "Batch Size ${BATCH_SIZE}" 
        echo "Num in context ${NUM}" 
        python -m privacy.main \
            --dataset sent140 \
            --model ${MODEL} \
            --paradigm prompt \
            --split test \
            --batch_size ${BATCH_SIZE} \
            --client_subsample 0.025 \
            --use_gpu 1 \
            --seed 0 \
            --max_sequence_length 512 \
            --prompt_choice random_incontext \
            --num_incontext ${NUM}
    done
done


# Command for GPT Variations with "No User Privacy" In context examples
for MODEL in gpt2.7 gpt1.3 gpt125m;  
do
    for NUM in 3 5; 
    do
        if [[ $MODEL == "gpt1.3" ]]; then
            BATCH_SIZE=32
        elif [[ $MODEL == "gpt125m" ]]; then 
            BATCH_SIZE=72
        elif [[ $MODEL == "gpt2.7" ]]; then 
            BATCH_SIZE=24
        fi
        
        echo "Model ${MODEL}" 
        echo "Batch Size ${BATCH_SIZE}" 
        echo "Num in context ${NUM}" 
        python -m privacy.main \
            --dataset sent140 \
            --model ${MODEL} \
            --paradigm prompt \
            --split test \
            --batch_size ${BATCH_SIZE} \
            --client_subsample 0.025 \
            --use_gpu 1 \
            --seed 0 \
            --max_sequence_length 512 \
            --prompt_choice random_incontext_noprivacy \
            --num_incontext ${NUM}
    done
done


# Command for GPT OpenAI API Inference 
python -m privacy.main \
    --dataset sent140 \
    --model gpt6.7 \
    --paradigm prompt \
    --split test \
    --batch_size 1 \
    --client_subsample 0.025 \
    --seed 0 \
    --prompt_choice random_incontext \
    --num_incontext 3 \
    --openai_key "fill in"
