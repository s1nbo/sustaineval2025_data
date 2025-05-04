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

for a_label in range(1, 21):
    word_count = defaultdict(int)
    for file in data:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                    line = json.loads(line)
                    if int(line['task_a_label'] == a_label):
                        task_a_label.append(int(line['task_a_label']))
                        task_b_label.append(round(float(line['task_b_label']), 4))
                        year.append(int(line['year']))
                        
                        for i in range(len(line['context'])):
                            line['context'][i] = line['context'][i].strip('"\'/.,;:!?()[]{}<>-=+*^&%$#@!~`|\\')
                            for word in re.findall(pattern, line['context'][i]):
                                if len(word) > 7:
                                    word_count[word] += 1
                        
                        target = line['target'].strip('"\'/.,;:!?()[]{}<>-=+*^&%$#@!~`|\\')
                        for word in re.findall(pattern, target):
                            if len(word) > 7:
                                    word_count[word] += 1
                            

    eval.save_word_count(word_count, 1, a_label)
                
