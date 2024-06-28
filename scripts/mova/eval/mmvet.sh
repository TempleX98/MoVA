#!/bin/bash

python -m mova.eval.model_vqa \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/mova-8b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/mova-8b.json

