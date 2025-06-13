from transformers import pipeline
import pandas as pd
import os
import json

# Paths
base_dir = os.path.join(os.path.dirname(__file__), '..')
data_path = os.path.join(base_dir, 'data')

# Load translation pipelines
de2en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
en2de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")


data_files = ['trial', 'training','development']

output = os.path.join(data_path, "generated_data.jsonl")

with open(output, 'w', encoding='utf-8') as outfile:
    for file_name in data_files:
        file_path = os.path.join(data_path, file_name+'_data.jsonl')
        current_file = pd.read_json(file_path, lines=True)

        for _, row in current_file.iterrows():

            # iterate through every row of current file and apply this and write to generated data
            context = ' '.join(row['context']) if isinstance(row['context'], list) else row['context']
            context += " " + row['target']

            # Take first context from current_file
            en_text = de2en(context, max_length=512)[0]['translation_text']
            back_text = en2de(en_text, max_length=512)[0]['translation_text']
            d = '.'
            context_list = [temp+d for temp in back_text.split(d) if temp]

            json_line = {
                "context": context_list,
                "task_a_label": row["task_a_label"]
            }

            outfile.write(json.dumps(json_line, ensure_ascii=False) + "\n")
