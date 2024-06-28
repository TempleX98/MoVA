import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, default='')
args = parser.parse_args()


indata = []

with open(args.filepath, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        indata.append(data)

total_num = 0.0
correct_num = 0.0

for item in indata:
    total_num += 1.0
    choice = ord(item['response'][0]) - ord('A')
    # print(choice)
    answer = item['correct_ans']
    try:
        if choice >= 0 and choice < len(item['candidates']) and item['candidates'][choice] == answer:
            correct_num += 1.0
        elif item['response'].split(' ')[0] == answer:
            correct_num += 1.0
        elif item['response'] == answer:
            correct_num += 1.0
    except Exception as e:
        print(e)
        print(item['response'])

print('#' * 50)
print(correct_num, total_num)
print(correct_num/total_num)

