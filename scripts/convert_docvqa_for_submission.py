import os
import json
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    f = open(os.path.join(args.result_dir, f"{args.experiment}.jsonl"), 'r')
    results = f.readlines()
    upload_results = []
    for pred in results:
        pred = json.loads(pred)
        upload_results.append({'questionId':pred['question_id', 'answer':pred['text']})
    f = open(os.path.join(args.upload_dir, f"{args.experiment}.json"), 'w')
    json.dump(upload_results, f)
