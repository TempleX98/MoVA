#!/bin/bash

python -m mova.eval.model_vqa_science \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/mova-8b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode mova_llama3

python mova/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/mova-8b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/mova-8b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/mova-8b_result.json
