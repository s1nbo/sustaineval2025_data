import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import wandb
import optuna
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, BertTokenizer,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)


class Model:
    '''
    Set Paramters and load data
    '''
    def __init__(self, target = 'validation'): # check if this hast to be trial or validation or whatever 
        self.setup()
       
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

        self.load_data()


    def load_data(self, target = 'validation', top_class: bool = False):
        # Load Data, we want to combine training and trial for training
        self.training = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.submission = pd.DataFrame() if not top_class else None
        self.data_files = ['trial', 'training','development', target]
        if top_class: self.data_files = self.data_files[:-1] 

        for file_name in self.data_files:
            # Read data from jsonl files
            file_path = os.path.join(self.data_path, file_name+'_data.jsonl')
            if not os.path.exists(file_path): raise FileNotFoundError(f'The file does not exist: {file_path}')
            current_file = pd.read_json(file_path, lines=True)
            
            # Convert list to string if the column is a list
            current_file['context'] = current_file['context'].apply(lambda x : ' '.join(x) if isinstance(x, list) else x)
            current_file['context'] += current_file['target']
            # change task_a_label to be zero-indexed
            if 'task_a_label' in current_file.columns:
                current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : x-1 if isinstance(x, int) else x)
                # for top level class
                if top_class:
                    current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : self.top_level_labels[x])

            if file_name == target:
                self.submission = current_file
            elif file_name == 'development':
                self.validation = current_file
            else:
                self.training = pd.concat([self.training, current_file], ignore_index=True)

    def setup(self):
        # Directory for results, should work on any os (Needs to be tested)
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
        self.epochs = 10             # How many epochs to train
        self.learning_rate = 0.00007336182194696851   # Learning rate for the optimizer, smaller = more stable
        self.weight_decay = 0.25209871615317314    # L2-regularization, to prevent overfitting
        self.batch_size = 16
        self.warmup_ratio = 0.2627025280744802


        '''
        self.pretrained_model_name = 'deepset/gbert-base'
        self.epochs = 6             # How many epochs to train
        self.learning_rate =  8.296246711509123e-05   # Learning rate for the optimizer, smaller = more stable
        self.weight_decay = 0.1653507912146719    # L2-regularization, to prevent overfitting
        self.batch_size = 4
        self.warmup_ratio = 0.049362010462963166
        '''


    def train_model(self, test = False):
        # Create Hugging Face Dataset        
        train_dataset = HFDataset.from_pandas(self.training[['context', 'task_a_label']])
        vali_dataset = HFDataset.from_pandas(self.validation[['context', 'task_a_label']])

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def tokenize_sample(example):
            return self.tokenizer(example['context'], truncation=True)

        tokenized_train = train_dataset.map(tokenize_sample, batched=True)
        tokenized_vali = vali_dataset.map(tokenize_sample, batched=True)


        tokenized_train = tokenized_train.rename_column('task_a_label', 'labels')
        tokenized_vali = tokenized_vali.rename_column('task_a_label', 'labels')
        tokenized_train.set_format('torch')
        tokenized_vali.set_format('torch')

        # Model preparation
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name, 
            num_labels = len(self.label_name)
        )

        # Define Training Arguments
        training_args = TrainingArguments(
            output_dir = self.result_path,                
            eval_strategy = 'epoch',
            save_strategy="epoch",              
            report_to = 'wandb',
            logging_dir="./logs",
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio
        )

        # Trainer Object
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_vali,
            data_collator=data_collator,
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))
            },
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
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
    def evaluate_model(self):

        model = AutoModelForSequenceClassification.from_pretrained(self.model_directory)
        model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)

        # Development Dataset
        dev_texts =  self.validation['context'].tolist()
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
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)

                for pred, prob in zip(preds, probs):
                    predictions.append((pred.item(), prob[pred].item()))

        # Save predictions to DataFrame
        self.validation['predicted_label'] = [p[0] for p in predictions]
        # self.training['confidence_score'] = [p[1] for p in predictions]
        
        # Calculate and save metrics
        y_true =  self.validation['task_a_label']
        y_pred =  self.validation['predicted_label']

        # Accuracy 
        acc = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {acc:.4f}')

        # Classification Report
        with open(os.path.join(self.result_path, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(classification_report(y_true=y_true, y_pred=y_pred, target_names=[self.label_name[i] for i in range(len(self.label_name))]))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(14,12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[self.label_name[i] for i in range(len(self.label_name))], yticklabels=[self.label_name[i] for i in range(len(self.label_name))])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix.png'))

        # Document Hyperparamters
        with open(os.path.join(self.result_path, 'parameters.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Name: {self.pretrained_model_name}\n')
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'Epochs: {self.epochs}\n')
            f.write(f'Learning rate: {self.learning_rate}\n')
            f.write(f'Weight decay: {self.weight_decay}\n')


    def generate_submission(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_directory)
        model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)

        # Submission Data
        dev_texts =  self.submission['context'].tolist()
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
        self.submission['predicted_label'] = [p[0] + 1 for p in predictions]

        # Save the predictions to a CSV file
        with open(os.path.join(self.result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
            f.write('id,label\n')
            for prediction in self.submission.iterrows():
                f.write (f"{prediction[1]['id']},{prediction[1]['predicted_label']}\n")

    def optuna_training(self, n_trials=20):
        '''
        Uses Optuna training instead of WandB sweep training.
        '''
        def objective(trial):
            # Suggest hyperparameters
            self.pretrained_model_name = trial.suggest_categorical(
                "model_name", 
                [
                'deepset/gbert-base', 'bert-base-german-cased', 'deepset/gelectra-base',
                'xlm-roberta-base', 'german-nlp-group/electra-base-german-uncased', 'bert-base-multilingual-cased'
                ]
            ) 
            self.learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-3)
            self.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.4)
            self.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
            self.epochs = trial.suggest_int("epochs", 6, 14)
            self.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)

            wandb.init(
                project="sustaineval2",
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "warmup_ratio": self.warmup_ratio,
                    "model_name": self.pretrained_model_name
                },
                reinit=True
            )

            # Tokenization & Dataset prep
            train_dataset = HFDataset.from_pandas(self.training[['context', 'task_a_label']])
            vali_dataset = HFDataset.from_pandas(self.validation[['context', 'task_a_label']])
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)

            def tokenize_sample(example):
                return self.tokenizer(example['context'], truncation=True)

            tokenized_train = train_dataset.map(tokenize_sample, batched=True)
            tokenized_vali = vali_dataset.map(tokenize_sample, batched=True)

            tokenized_train = tokenized_train.rename_column('task_a_label', 'labels')
            tokenized_vali = tokenized_vali.rename_column('task_a_label', 'labels')
            tokenized_train.set_format('torch')
            tokenized_vali.set_format('torch')

            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            model_for_training = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_name,
                num_labels=len(self.label_name)
            )

            training_args = TrainingArguments(
                output_dir=self.result_path,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                weight_decay=self.weight_decay,
                report_to="wandb",
                logging_dir="./logs",
                disable_tqdm=True,
                warmup_ratio=self.warmup_ratio,
                load_best_model_at_end=True,  
                metric_for_best_model="accuracy",
                greater_is_better=True
            )

            trainer = Trainer(
                model=model_for_training,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_vali,
                data_collator=data_collator,
                compute_metrics=lambda p: {
                    "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))
                },
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            trainer.train()
            eval_result = trainer.evaluate()
            wandb.log(eval_result)
            wandb.finish()
            return eval_result["eval_accuracy"]

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:")
        print(study.best_trial)

        # Update model with best parameters
        best_params = study.best_trial.params
        self.learning_rate = best_params["learning_rate"]
        self.weight_decay = best_params["weight_decay"]
        self.batch_size = best_params["batch_size"]
        self.epochs = best_params["epochs"]
        self.warmup_ratio = best_params["warmup_ratio"]

        # Train final model with best hyperparameters
        print("Training final model with best hyperparameters...")
        self.train_model()
        self.evaluate_model()
        self.generate_submission()


if __name__ == "__main__":
    model = Model()
    model.train_model()
    model.evaluate_model()
    model.generate_submission()


