#!/bin/bash

python -m mova.eval.model_vqa \
    --model-path liuhaotian/mova-8b \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python mova/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule mova/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/mova-8b.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/mova-8b.jsonl

python mova/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/mova-8b.jsonl
