#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m mova.eval.model_vqa_mmbench \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/mova-8b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode mova_llama3

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment mova-8b
