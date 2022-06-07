# Command for GPT Variations with "User Privacy" In context examples
for MODEL in gpt2.7 gpt1.3 gpt125m;  
do
    for NUM in 3 5; 
    do
        if [[ $MODEL == "gpt1.3" ]]; then
            BATCH_SIZE=64
        elif [[ $MODEL == "gpt125m" ]]; then 
            BATCH_SIZE=128
        elif [[ $MODEL == "gpt2.7" ]]; then 
            BATCH_SIZE=32
        fi
        
        echo "Model ${MODEL}" 
        echo "Batch Size ${BATCH_SIZE}" 
        echo "Num in context ${NUM}" 
        python -m privacy.main \
            --dataset reddit \
            --model ${MODEL} \
            --paradigm prompt \
            --split test \
            --batch_size ${BATCH_SIZE} \
            --use_gpu 1 \
            --seed 0 \
            --max_sequence_length 324 \
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
            BATCH_SIZE=64
        elif [[ $MODEL == "gpt125m" ]]; then 
            BATCH_SIZE=128
        elif [[ $MODEL == "gpt2.7" ]]; then 
            BATCH_SIZE=36
        fi
        
        echo "Model ${MODEL}" 
        echo "Batch Size ${BATCH_SIZE}" 
        echo "Num in context ${NUM}" 
        python -m privacy.main \
            --dataset reddit \
            --model ${MODEL} \
            --paradigm prompt \
            --split test \
            --batch_size ${BATCH_SIZE} \
            --use_gpu 1 \
            --seed 0 \
            --max_sequence_length 324 \
            --prompt_choice random_incontext_noprivacy \
            --num_incontext ${NUM}
    done
done


# Command for GPT OpenAI API Inference 
python -m privacy.main \
    --dataset reddit \
    --model gpt6.7 \
    --paradigm prompt \
    --split test \
    --batch_size 1 \
    --seed 0 \
    --prompt_choice random_incontext \
    --num_incontext 1 \
    --openai_key "fill in"