#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/chartqa/test_all.jsonl \
    --image-folder ./playground/data/eval/chartqa \
    --answers-file ./playground/data/eval/chartqa/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

python ./playground/data/eval/chartqa/eval_chartqa.py \
    --annotation-file ./playground/data/eval/chartqa/test_all.jsonl \
    --result-file ./playground/data/eval/chartqa/answers/mova-8b.jsonl \
    --mid_result ./playground/data/eval/chartqa/mid_results/mova-8b.jsonl \
    --output_result ./playground/data/eval/chartqa/results/mova-8b.jsonl 
