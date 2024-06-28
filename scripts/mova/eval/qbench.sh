#!/bin/bash

if [ "$1" = "dev" ]; then
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

python -m mova.eval.model_vqa_qbench \
    --model-path checkpoints/mova-8b \
    --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/llvisionqa_$1.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_$1_answers.jsonl \
    --conv-mode mova_llama3 \
    --lang en
