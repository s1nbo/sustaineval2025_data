import random
import json

# get the id from validation data
with open('../data/validation_data.jsonl', 'r') as f:
    lines = f.readlines()
    ids = []
    for line in lines:
        line = json.loads(line)
        ids.append(line['id'])
# create a random submission file with csv format:
#  906t, (random.randint(1, 20)
print(ids)
with open('prediction_task_a.csv', 'a') as f:
    f.write('id,label\n')
    for i in range(len(ids)):
        f.write(f'{ids[i]},{random.randint(1, 20)}\n')