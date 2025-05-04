import eval
import json
import re
from collections import defaultdict

data = ['data/trial_data.jsonl', 'data/development_data.jsonl', 'data/training_data.jsonl', 'data/validation_data.jsonl']
data = data[0:3]

task_a_label = []
task_b_label = []
word_count = defaultdict(int)
year = []

pattern = r'\b[a-z0-9äöüß]+\b' # regex pattern to match words

for file in data:
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            task_a_label.append(int(line['task_a_label']))
            task_b_label.append(round(float(line['task_b_label']), 4))
            year.append(int(line['year']))
            
            for i in range(len(line['context'])):
                line['context'][i] = line['context'][i].strip('"\'/.,;:!?()[]{}<>-=+*^&%$#@!~`|\\')
                for word in re.findall(pattern, line['context'][i]):
                    word_count[word] += 1
            
            target = line['target'].strip('"\'/.,;:!?()[]{}<>-=+*^&%$#@!~`|\\')
            for word in re.findall(pattern, target):
                word_count[word] += 1
            



            

            
# print the most common 10 words
# check if every key is a string
for key in word_count.keys():
    if not isinstance(key, str):
        print(f'Key {key} is not a string')
        break
# give lenght of word_count
print(f'Length of word_count: {len(word_count)}')
# give length of word_count with more than 1 occurence
print(len([key for key in word_count.keys() if word_count[key] > 1]))



