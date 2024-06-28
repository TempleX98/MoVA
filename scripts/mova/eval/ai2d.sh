#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/ai2d/test.jsonl \
    --image-folder ./playground/data/eval/ai2d/ai2d_images \
    --answers-file ./playground/data/eval/ai2d/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

python ./playground/data/eval/ai2d/eval_ai2d.py \
    --annotation-file ./playground/data/eval/ai2d/test.jsonl \
    --result-file ./playground/data/eval/ai2d/answers/mova-8b.jsonl \
    --mid_result ./playground/data/eval/ai2d/mid_results/mova-8b.jsonl \
    --output_result ./playground/data/eval/ai2d/results/mova-8b.jsonl 
