import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, BertTokenizer,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer)
from sklearn.metrics import (accuracy_score, confusion_matrix)

class Model:
    '''
    Set Paramters and load data
    '''
    def __init__(self, target = "validation"): # check if this hast to be trial or validation or whatever 
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
        

        # Model Configuration / These Paramaters are set by Optuna training
        self.pretrained_model_name = 'bert-base-german-cased'
        self.training_steps = 500 # More steps = more time
        self.epochs = 8             # How many epochs to train
        self.learning_rate = 4.4e-5   # Learning rate for the optimizer, smaller = more stable
        self.weight_decay = 0.08    # L2-regularization, to prevent overfitting

        
        # Label Names
        self.label_name = {
            0: 'Strategic Analysis and Action',   1: 'Materiality',
            2: 'Objectives',                      3: 'Depth of the Value Chain',
            4: 'Responsibility',                  5: 'Rules and Processes',
            6: 'Control',                         7: 'Incentive Systems',
            8: 'Stakeholder Engagement',          9: 'Innovation and Product Management',
            10: 'Usage of Natural Resources',     11: 'Resource Management',
            12: 'Climate-Relevant Emissions',     13: 'Employment Rights',
            14: 'Equal Opportunities',            15: 'Qualifications',
            16: 'Human Rights',                   17: 'Corporate Citizenship',
            18: 'Political Influence',            19: 'Conduct that Complies with the Law and Policy'
        }

        # Load Data
        # X is the input data, Y is the target data
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.data_files = ['trial', 'training','development', target]


        for file_name in self.data_files:
            # Read data from jsonl files
            file_path = os.path.join(self.data_path, file_name+'_data.jsonl')
            if not os.path.exists(file_path): raise FileNotFoundError(f'The file does not exist: {file_path}')
            current_file = pd.read_json(file_path, lines=True)
            
            # Convert list to string if the column is a list
            current_file['context'] = current_file['context'].apply(lambda x : ' '.join(x) if isinstance(x, list) else x)
            # change task_a_label to be zero-indexed
            if 'task_a_label' in current_file.columns:
                current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : x-1 if isinstance(x, int) else x)
            
            if file_name == target:
                self.Y = current_file
            else:
                self.X = pd.concat([self.X, current_file], ignore_index=True)
        

    def train_auto_model(self, test = False):
        # Create Hugging Face Dataset        
        train_dataset = HFDataset.from_pandas(self.X[['context', 'task_a_label']])

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def tokenize_sample(example):
            return self.tokenizer(example['context'], truncation=True)

        tokenized_train = train_dataset.map(tokenize_sample, batched=True)

        if test:
            # For fast testing, select only a few samples
            tokenized_train = tokenized_train.select(range(5))

        tokenized_train = tokenized_train.rename_column('task_a_label', 'labels')
        tokenized_train.set_format('torch')


        # Model preparation
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels = len(self.label_name))
        
        # Define Training Arguments
        training_args = TrainingArguments(
            output_dir = self.result_path,                # Directory for saving the model
            eval_strategy = 'no',              
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
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator # What exactly is this?
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


    # Loads the model and tokenizer and evaluates the model on the given data
    # plotting can be moved to another function/class and evaluation can be returned after training TODO (function can be split up and removed)
    def evaluate_model(self, custom_model = False):

        model = AutoModelForSequenceClassification.from_pretrained(self.model_directory)
        model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)

        # Development Dataset
        dev_texts =  self.X['context'].tolist()
        dev_dataset = HFDataset.from_dict({'context': dev_texts})

        # Tokenizing
        def tokenize_batch(batch):
            return self.tokenizer(batch['context'], truncation=True)
        
        dev_dataset = dev_dataset.map(tokenize_batch, batched=True, remove_columns=['context'])

        # Use GPU or CPU, Why here and not above??
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=data_collator)

        # Predictions
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

        # Save predictions to DataFrame
        self.X['predicted_label'] = [p[0] for p in predictions]
        # self.X['confidence_score'] = [p[1] for p in predictions]
        
        # Calculate and save metrics
        y_true =  self.X['task_a_label']
        y_pred =  self.X['predicted_label']

        # Accuracy 
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


    def submission(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_directory)
        model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)

        # Real Data
        dev_texts =  self.Y['context'].tolist()
        dev_dataset = HFDataset.from_dict({'context': dev_texts})

        # Tokenizing
        def tokenize_batch(batch):
            return self.tokenizer(batch['context'], truncation=True)
        
        dev_dataset = dev_dataset.map(tokenize_batch, batched=True, remove_columns=['context'])

        # Use GPU or CPU
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=data_collator)

        # Predictions
        predictions = []

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)

                for pred, prob in zip(preds, probs):
                    predictions.append((pred.item(), prob[pred].item()))
        # Save predictions to DataFrame and add one
        self.Y['predicted_label'] = [p[0] + 1 for p in predictions]

        # Save the predictions to a CSV file
        with open(os.path.join(self.result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
            f.write('id,label\n')
            for prediction in self.Y.iterrows():
                f.write (f"{prediction[1]['id']},{prediction[1]['predicted_label']}\n")





model = Model()

print('TRAINING AUTO MODEL')
#model.train_auto_model()
print('EVALUATING AUTO MODEL')
#model.evaluate_model()
model.submission()

