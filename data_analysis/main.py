import data_analysis.eval as eval
import json
import re
from collections import defaultdict

data = ['data/trial_data.jsonl', 'data/development_data.jsonl', 'data/training_data.jsonl', 'data/validation_data.jsonl']
data = data[0:3]

task_a_label = []
task_b_label = []
possible_labels = [0.0, 0.1111, 0.1667, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0]
word_count = defaultdict(int)
year = []
length_of_data = defaultdict(list)


pattern = r'\b[a-z0-9äöüß]+\b' # regex pattern to match words

for label in range(1, 21):
    for file in data:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                    line = json.loads(line)
                    if int(label == line['task_a_label']):
                        task_a_label.append(int(line['task_a_label']))
                        task_b_label.append(round(float(line['task_b_label']), 4))
                        
                        year.append(int(line['year']))
                        
                        for i in range(len(line['context'])):
                            line['context'][i] = line['context'][i].strip('"\'/.,;:!?()[]{}<>-=+*^&%$#@!~`|\\')
                            for word in re.findall(pattern, line['context'][i]):
                                word_count[word] += 1
                        
                        '''
                        target = line['target'].strip('"\'/.,;:!?()[]{}<>-=+*^&%$#@!~`|\\')
                        for word in re.findall(pattern, target):
                            word_count[word] += 1
                        '''
                
                        length_of_data[label].append(sum([word_count[word] for word in word_count.keys()]))
                        word_count = defaultdict(int)


eval.length(length_of_data)
                        

                        

# Task B Labels: 0.8889, 0.5556, 0.2222, 1.0, 0.3333, 0.0, 0.6667, 0.8333, 0.5, 0.1667, 0.7778, 0.1111, 0.4444