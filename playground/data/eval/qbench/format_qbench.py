import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, default='')
args = parser.parse_args()

file = args.filepath
content = open(file).read()

content = content.replace('}{', '}\n{')

with open(file, 'w') as fout:
    fout.write(content)

