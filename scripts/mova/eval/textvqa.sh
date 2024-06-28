#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

python -m mova.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/mova-8b.jsonl
