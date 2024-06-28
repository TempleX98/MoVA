#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/docvqa/val.jsonl \
    --image-folder ./playground/data/eval/docvqa/val/documents/ \
    --answers-file ./playground/data/eval/docvqa/answers/mova-8b-val.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

python ./playground/data/eval/docvqa/eval_docvqa.py \
    --annotation-file ./playground/data/eval/docvqa/val.jsonl \
    --result-file ./playground/data/eval/docvqa/answers/mova-8b-val.jsonl \
    --mid_result ./playground/data/eval/docvqa/mid_results/mova-8b-val.jsonl \
    --output_result ./playground/data/eval/docvqa/results/mova-8b-val.jsonl 
