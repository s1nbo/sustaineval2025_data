from transformers import pipeline
import pandas as pd

# Load translation pipelines
de2en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
en2de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")

current_file = pd.read_json('trial_data.jsonl', lines=True)

current_file['context'] = current_file['context'].apply(lambda x : ' '.join(x) if isinstance(x, list) else x)
current_file['context'] += current_file['target']

# Take first context from current_file
temp = current_file['context'][0]
print(temp)
en_text = de2en(temp, max_length=512)[0]['translation_text']
back_text = en2de(en_text, max_length=512)[0]['translation_text']
print(back_text)
