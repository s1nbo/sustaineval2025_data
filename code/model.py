from collections import defaultdict
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, BertTokenizer,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)

class Model:
    '''
    Set Paramters and load data
    '''
    def __init__(self, trial = False): # check if this hast to be trial or validation or whatever 
        # Directory for results
        self.result_path = '../result'
        self.model_directory = os.path.join(self.result_path, 'checkpoints')
        self.parameter_file = os.path.join(self.result_path, 'parameters.txt')
        self.data_path = '../data'
        
        # Create and check directories
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_directory, exist_ok=True)
        # Check if the directories exist
        if not os.path.exists(self.result_path):
            raise FileNotFoundError(f'The directory does not exist: {self.result_path}')
        if not os.path.exists(self.model_directory):
            raise FileNotFoundError(f'The directory does not exist: {self.model_directory}')
        

        # Model Configuration
        self.pretrained_model_name = 'bert-base-german-cased'
        self.tokenizer = None # Will be set during training
        self.training_steps = 150 # More steps = more time
        self.relevant_words = 400 # Number of words to select for TF-IDF
        self.context_target_ratio = 1.1 # scaling of target vs context for evaluation

        # These Paramaters are set by Optuna training
        self.epochs = 8             # How many epochs to train
        self.learning_rate = 2e-5   # Learning rate for the optimizer, smaller = more stable
        self.weight_decay = 0.01    # L2-regularization, to prevent overfitting
        self.warmup_ratio = 0.1     # Warmup ratio for the learning rate scheduler
        self.save_current_parameters(self.parameter_file)
        
        # Label Names
        # Watch out I've updated the Keys to start at 1.
        self.label_name = { 
            1: 'Strategic Analysis and Action',   2: 'Materiality',
            3: 'Objectives',                      4: 'Depth of the Value Chain',
            5: 'Responsibility',                  6: 'Rules and Processes',
            7: 'Control',                         8: 'Incentive Systems',
            9: 'Stakeholder Engagement',         10: 'Innovation and Product Management',
            11: 'Usage of Natural Resources',    12: 'Resource Management',
            13: 'Climate-Relevant Emissions',    14: 'Employment Rights',
            15: 'Equal Opportunities',           16: 'Qualifications',
            17: 'Human Rights',                  18: 'Corporate Citizenship',
            19: 'Political Influence',           20: 'Conduct that Complies with the Law and Policy'
        }

        # DO WE NEED THIS? TODO
        self.label_superlabel = {
            1: 0, 2: 0, 3: 0, 4: 0,                             # 0 Strategy
            5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,                # 1 Process Management
            11: 2, 12: 2, 13: 2,                                # 2 Environment
            14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3     # 3 Society
        }

        
        # Load Data
        self.data_files = ['trial' if trial else 'training','validation','development']
        self.data = defaultdict(pd.DataFrame)


        for file_name in self.data_files:
            # Read data from jsonl files
            file_path = os.path.join(self.data_path, file_name+'_data.jsonl')
            if not os.path.exists(file_path): raise FileNotFoundError(f'The file does not exist: {file_path}')
            current_file = pd.read_json(file_path, lines=True)
            
            # Prepare data for training
            # change context from list to string
            current_file['context'] = current_file['context'].apply(lambda x : ' '.join(x) if isinstance(x, list) else x)
            self.data[file_name] = (current_file)


    # Trains Bert-Automodel with validation datat without labels
    def train_auto_model(self):
        with open(self.parameter_file, 'a') as file:
            file.write("\nmodel = auto\n")
        
        # Create Hugging Face Dataset        
        train_dataset = HFDataset.from_pandas(self.data['training'][['context', 'task_a_label']])
        val_dataset = HFDataset.from_pandas(self.data['validation'][['context']])

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def tokenize_sample(example):
            return self.tokenizer(example['context'], truncation=True)

        tokenized_train = train_dataset.map(tokenize_sample, batched=True).set_format('torch')
        tokenized_val = val_dataset.map(tokenize_sample, batched=True).set_format('torch')


        # Model preparation
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=len(self.label_name))
        
        # Define Training Arguments
        training_args = TrainingArguments(
            output_dir = self.result_path,                # Directory for saving the model
            eval_strategy = 'steps',              
            eval_steps = self.training_steps,             # After how many steps should be evaluated
            save_steps = self.training_steps,             # After how many steps should the model be saved
            logging_steps = int(self.training_steps*1/5), # After how many steps should be logged
            num_train_epochs = self.epochs,               # Number of epochs to train
            learning_rate = self.learning_rate,   
            weight_decay = self.weight_decay,
            save_total_limit = 1,
            report_to = 'none'                            # No external tracking services
        )

        # Trainer Object
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = tokenized_train,
            eval_dataset = tokenized_val,
            data_collator = data_collator
        )

        # Training
        start_time = time.time()
        print('Start Training')
        trainer.train()
        end_time = time.time()
        training_duration = end_time - start_time
        print(f'Finished Training in {training_duration:.4f} seconds.')

        # Save Model
        trainer.save_model(self.model_directory)
        self.tokenizer.save_pretrained(self.model_directory)

    # TODO        
    # Läd Model aus Results-Pfad, evaluiert Model mit Development-Daten, self.loged Klassifikationsbericht in Konsole und speichert Confusion-Matrix in Results-Pfad
    def evaluate_model(self, custom_model = False):

        print('Evaluating Model')
        model = AutoModelForSequenceClassification.from_pretrained(self.model_directory)
        model.eval()

        print('Loading Tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)

        # Development Dataset
        dev_texts =  self.data['development']['context'].tolist()
        dev_dataset = HFDataset.from_dict({'context': dev_texts})

        # Tokenizing
        def tokenize_batch(batch):
            return self.tokenizer(batch['context'], truncation=True)
        
        dev_dataset = dev_dataset.map(tokenize_batch, batched=True, remove_columns=['context'])

        # Use GPU
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cuda')
        model.to(device)

        # DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=data_collator)

        # Vorhersagen
        predictions = []

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                if custom_model:
                    probs = torch.nn.functional.softmax(outputs['logits'], dim=-1)
                else:
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)

                for pred, prob in zip(preds, probs):
                    predictions.append((pred.item(), prob[pred].item()))

        # Ergebnisse in DataFrame speichern
        self.data['development']['predicted_label'] = [p[0] for p in predictions]
        self.data['development']['confidence_score'] = [p[1] for p in predictions]

        # Schöne Labels ergänzen
        self.data['development']['true_label_name'] =  self.data['development']['label_name']
        self.data['development']['predicted_label_name'] =  self.data['development']['predicted_label'].map(self.label_name)

        # Excel speichern
        self.data['development'][['id', 'year', 'context', 
                             'true_label_name', 'predicted_label_name', 
                             'confidence_score']].to_excel(self.result_path + '\\development_predictions.xlsx', index=False)

        print(f'''Vorhersagen gespeichert unter: {self.result_path} - development_predictions.xlsx''')

        # Metriken berechnen
        y_true =  self.data['development']['task_a_label']
        y_pred =  self.data['development']['predicted_label']

        '''
        # Classification Report for all classes Maybe we need this
        
        report_dict = classification_report(y_true, y_pred, target_names=[self.label_name[i] for i in range(20)], 
                                            digits=3, output_dict=True)

        df_report = pd.DataFrame(report_dict).transpose()
        '''


        # Genauigkeit berechnen
        acc = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {acc:.4f}')

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(14,12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[self.label_name[i] for i in range(20)], yticklabels=[self.label_name[i] for i in range(20)])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix.png'))
        plt.close()

    ###########################################################
    # Hilfs-Funktionen
    ###########################################################


    def save_current_parameters(self, filepath):
        # Manuell festgelegte Parameter, die du loggen willst – alle aus der Instanz
        tracked_params = [
            "training_steps",
            "relevant_words",
            "context_target_ratio",
            "epochs",
            "learning_rate",
            "weight_decay",
            "warmup_ratio"
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            for param in tracked_params:
                value = getattr(self, param, None)
                line = f"{param} = {value}"
                f.write(line + '\n')


    def optuna_hp_space(self, trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            'num_train_epochs': trial.suggest_int('num_train_epochs', 3, 10),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32]),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1)
        }

    def compute_metrics(self, eval_pred):
        '''Funktion num Metriken in Training zu integrieren.'''
        # Überprüfen, ob predictions ein Tuple sind
        if isinstance(eval_pred.predictions, tuple):
            # Wenn ja, nimm nur die erste Prediction (label_logits)
            logits = eval_pred.predictions[0]
        else:
            # Wenn nein, normale Prediction
            logits = eval_pred.predictions

        preds = np.argmax(logits, axis=1)
        labels = eval_pred.label_ids

        # Sicherstellen, dass labels ein Array sind
        if isinstance(labels, tuple):
            labels = labels[0]

        return {
            'eval_accuracy': accuracy_score(labels, preds),
            'eval_f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'eval_precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'eval_recall_macro': recall_score(labels, preds, average='macro', zero_division=0)
        }