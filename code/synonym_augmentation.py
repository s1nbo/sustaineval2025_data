import random
import nltk
from nltk.corpus import wordnet
from transformers import pipeline
import pandas as pd
import os
import json

def get_synonyms(word):
    # Get a list of synonyms for a word using WordNet
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Avoid the original word itself
            if lemma.name().lower() != word.lower():
                # Replace underscores with spaces (multi-word synonyms)
                synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)


def synonym_replacement(sentence, n_replacements=3):
    words = sentence.split()
    if len(words) == 0:
        return sentence

    indices = list(range(len(words)))
    random.shuffle(indices)

    replaced = 0
    for idx in indices:
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            # Replace with a random synonym
            words[idx] = random.choice(synonyms)
            replaced += 1
            if replaced >= n_replacements:
                break

    return ' '.join(words)


if __name__ == "__main__":

    nltk.download('wordnet') # Necessary 

    # Read Data
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_dir, 'data')

    data_files = ['trial', 'training']

    output = os.path.join(data_path, "synonym_data.jsonl")

    with open(output, 'w', encoding='utf-8') as outfile:
        for file_name in data_files:
            file_path = os.path.join(data_path, file_name+'_data.jsonl')
            current_file = pd.read_json(file_path, lines=True)
        

        # iterate through every row of current file and apply synonym_replacement
        for _, row in current_file.iterrows():
            context = ' '.join(row['context']) if isinstance(row['context'], list) else row['context']
            context += " " + row['target']

            generated_data = synonym_replacement(context, n_replacements=7)

            # Write to output
            d = '.'
            context_list = [temp+d for temp in generated_data.split(d) if temp]
            json_line = {
                "context": context_list,
                "task_a_label": row["task_a_label"]
            }

            outfile.write(json.dumps(json_line, ensure_ascii=False) + "\n")