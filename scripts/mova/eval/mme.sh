#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment mova-8b

cd eval_tool

python calculation.py --results_dir answers/mova-8b
