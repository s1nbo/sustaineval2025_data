import os
import time
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, BertTokenizer,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer)
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

from model import Model


class TopLevel(Model):
    def __init__(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), '..')

        self.result_path = os.path.join(self.base_dir, 'result')
        self.model_directory = os.path.join(self.result_path, 'checkpoints')
        self.parameter_file = os.path.join(self.result_path, 'parameters.txt')
        self.data_path = os.path.join(self.base_dir, 'data')
        
        # Create and check directories
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_directory, exist_ok=True)
        # Check if the directories exist
        if not os.path.exists(self.result_path):
            raise FileNotFoundError(f'The directory does not exist: {self.result_path}')
        if not os.path.exists(self.model_directory):
            raise FileNotFoundError(f'The directory does not exist: {self.model_directory}')
        
        # Model Configuration / These Paramaters are set by Optuna training
        self.pretrained_model_name = 'deepset/gbert-base'
        self.training_steps = 500 # More steps = more time
        self.epochs = 8             # How many epochs to train
        self.learning_rate = 4.4e-5   # Learning rate for the optimizer, smaller = more stable
        self.weight_decay = 0.08    # L2-regularization, to prevent overfitting


        self.label_name = [0, 1, 2, 3]
        # Label Names
        self.top_level_labels = {
             0: 0,  1: 0,
             2: 0,  3: 0,
             4: 1,  5: 1,
             6: 1,  7: 1,
             8: 1,  9: 1,
            10: 2, 11: 2,
            12: 2, 13: 3,
            14: 3, 15: 3,
            16: 3, 17: 3,
            18: 3, 19: 3
        }

        # Load Data, we want to combine training and trial for training
        self.training = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.data_files = ['trial', 'training','development']

        for file_name in self.data_files:
            # Read data from jsonl files
            file_path = os.path.join(self.data_path, file_name+'_data.jsonl')
            if not os.path.exists(file_path): raise FileNotFoundError(f'The file does not exist: {file_path}')
            current_file = pd.read_json(file_path, lines=True)
            
            # Convert list to string if the column is a list
            current_file['context'] = current_file['context'].apply(lambda x : ' '.join(x) if isinstance(x, list) else x)
            # change task_a_label to Toplevel class label
            if 'task_a_label' in current_file.columns:
                current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : x-1 if isinstance(x, int) else x)
                current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : self.top_level_labels[x])
            
            if file_name == 'development':
                self.validation = current_file
            else:
                self.training = pd.concat([self.training, current_file], ignore_index=True)


if __name__ == '__main__':
    t = TopLevel()
    t.train_auto_model()
    t.evaluate_model()
