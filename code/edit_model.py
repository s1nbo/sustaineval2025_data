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
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)


class ModelEdit:
    '''
    Set Paramters and load data
    '''
    def __init__(self): # check if this hast to be trial or validation or whatever 
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

        self.super_label_name = {
            0: 'Strategy',
            1: 'Process Management',
            2: 'Environment',
            3: 'Society'
        }

        # maps labels to super labels
        self.label_to_superlabel = {
            0: 0, 1: 0, 2: 0, 3: 0,                             # 0 Strategy
            4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,                 # 1 Process Management
            10: 2, 11: 2, 12: 2,                                # 2 Environment
            13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3     # 3 Society
        }

        self.load_data()


    def load_data(self, target = 'evaluation', top_class: bool = True):
        # Load Data, we want to combine training and trial for training
        self.training = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.submission = pd.DataFrame() #if not top_class else None
        self.data_files = ['trial', 'training','development', 'generated', target]
        #if top_class: self.data_files = self.data_files[:-1] 

        for file_name in self.data_files:
            # Read data from jsonl files
            file_path = os.path.join(self.data_path, file_name+'_data.jsonl')
            if not os.path.exists(file_path): raise FileNotFoundError(f'The file does not exist: {file_path}')
            current_file = pd.read_json(file_path, lines=True)
            
            # Convert list to string if the column is a list
            current_file['context'] = current_file['context'].apply(lambda x : ' '.join(x) if isinstance(x, list) else x)
            
            if 'target' in current_file and current_file['target'] is not None:
                current_file['context'] += current_file['target']
            
            # change task_a_label to be zero-indexed
            if 'task_a_label' in current_file.columns:
                current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : x-1 if isinstance(x, int) else x)
                # for top level class
                if top_class:
                    current_file['super_label'] = current_file['task_a_label'].apply(lambda x : self.label_to_superlabel[x])

            if file_name == target:
                self.submission = current_file
            elif file_name == 'development':
                self.validation = current_file
            else:
                self.training = pd.concat([self.training, current_file], ignore_index=True)

    def setup(self):
        # Directory for results, should work on any os (Needs to be tested)
        self.base_dir = os.path.join(os.path.dirname(__file__), '..')
        self.result_dir = os.path.join(os.path.dirname(__file__), '.')

        self.result_path = os.path.join(self.result_dir, 'result')
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
        self.epochs = 9             # How many epochs to train
        self.learning_rate = 0.00004818995940467737   # Learning rate for the optimizer, smaller = more stable
        self.weight_decay = 0.2767731286383088   # L2-regularization, to prevent overfitting
        self.batch_size = 16
        self.warmup_ratio = 0.26868450115020465
        

    def train_model(self):
        # Create Hugging Face Dataset        
        train_dataset = HFDataset.from_pandas(self.training[['context', 'task_a_label', 'super_label']])
        vali_dataset = HFDataset.from_pandas(self.validation[['context', 'task_a_label', 'super_label']])

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def tokenize_sample(example):
            return self.tokenizer(example['context'], truncation=True)

        tokenized_train = train_dataset.map(tokenize_sample, batched=True)
        tokenized_vali = vali_dataset.map(tokenize_sample, batched=True)

        tokenized_train = tokenized_train.rename_column('task_a_label', 'labels')
        tokenized_vali = tokenized_vali.rename_column('task_a_label', 'labels')
        tokenized_train = tokenized_train.rename_column('super_label', 'super_labels')
        tokenized_vali = tokenized_vali.rename_column('super_label', 'super_labels')
        tokenized_train.set_format('torch')
        tokenized_vali.set_format('torch')

        model = AutoModel_manual(
            pretrained=self.pretrained_model_name,
            labels=len(self.label_name),
            super_labels=len(self.super_label_name)
        )

        wandb.init(project='top_level_sirak')

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
                "accuracy": accuracy_score(p.label_ids[0], p.predictions.argmax(-1))
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
        torch.save(model.state_dict(), os.path.join(self.model_directory, "model.pt"))
        self.tokenizer.save_pretrained(self.model_directory)
    
    # Loads the model and tokenizer and evaluates the model on the given data
    def evaluate_model(self, ensamble: bool = False):

        model = AutoModel_manual(
            pretrained=self.pretrained_model_name,
            labels=len(self.label_name),
            super_labels=len(self.super_label_name)
        )
        model.load_state_dict(torch.load(os.path.join(self.model_directory, "model.pt")))
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
        # self.validation['confidence_score'] = [p[1] for p in predictions]
        
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
            f.write(f'Batch size: {self.batch_size}\n')
            f.write(f'Weight decay: {self.weight_decay}\n')
            f.write(f'Warmup ratio: {self.warmup_ratio}\n')

        if ensamble:
            return y_pred

    def generate_submission(self, ensamble: bool = False):
        model = AutoModel_manual(
            pretrained=self.pretrained_model_name,
            labels=len(self.label_name),
            super_labels=len(self.super_label_name)
        )
        model.load_state_dict(torch.load(os.path.join(self.model_directory, "model.pt")))
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

        if ensamble:
            return self.submission['predicted_label']


        # Save the predictions to a CSV file
        with open(os.path.join(self.result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
            f.write('id,label\n')
            for prediction in self.submission.iterrows():
                f.write (f"{prediction[1]['id']},{prediction[1]['predicted_label']}\n")

    def optuna_training(self, n_trials=20, wandb_project="sustaineval"):
        '''
        Uses Optuna training instead of WandB sweep training.
        '''
        def objective(trial):
            # Suggest hyperparameter ranges
            # deepset/gbert-base superior model
            #self.pretrained_model_name = trial.suggest_categorical("model_name", ['deepset/gbert-base', 'bert-base-german-cased'])
            self.pretrained_model_name = 'deepset/gbert-base'
            self.learning_rate = trial.suggest_float("learning_rate", 0.00002, 0.00015)
            self.weight_decay = trial.suggest_float("weight_decay", 0.12, 0.35)
            self.batch_size = trial.suggest_categorical("batch_size", [4, 32])
            self.batch_size = 16
            self.epochs = trial.suggest_int("epochs", 8, 12)
            self.warmup_ratio = trial.suggest_float("warmup_ratio", 0.23, 0.35)

            wandb.init(
                project=wandb_project,
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "warmup_ratio": self.warmup_ratio,
                    # "model_name": self.pretrained_model_name
                },
                reinit=True
            )

            # Tokenization & Dataset prep
            train_dataset = HFDataset.from_pandas(self.training[['context', 'task_a_label', 'super_label']])
            vali_dataset = HFDataset.from_pandas(self.validation[['context', 'task_a_label', 'super_label']])
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=False)

            def tokenize_sample(example):
                return self.tokenizer(example['context'], truncation=True)

            tokenized_train = train_dataset.map(tokenize_sample, batched=True)
            tokenized_vali = vali_dataset.map(tokenize_sample, batched=True)

            tokenized_train = tokenized_train.rename_column('task_a_label', 'labels')
            tokenized_vali = tokenized_vali.rename_column('task_a_label', 'labels')
            tokenized_train = tokenized_train.rename_column('super_label', 'super_labels')
            tokenized_vali = tokenized_vali.rename_column('super_label', 'super_labels')
            tokenized_train.set_format('torch')
            tokenized_vali.set_format('torch')

            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            model_for_training = AutoModel_manual(
                pretrained=self.pretrained_model_name,
                labels=len(self.label_name),
                super_labels=len(self.super_label_name)
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
                    "accuracy": accuracy_score(p.label_ids[0], p.predictions.argmax(-1))
                },
                callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),
                EarlyStoppingWandbLoggerCallback()
                ]
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

        # Not sure if this works
        # Update model with best parameters
        best_params = study.best_trial.params
        self.learning_rate = best_params["learning_rate"]
        self.weight_decay = best_params["weight_decay"]
        self.epochs = best_params["epochs"]
        self.warmup_ratio = best_params["warmup_ratio"]

        with open(os.path.join(self.result_path, 'parameters.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Name: {self.pretrained_model_name}\n')
            f.write(f'Epochs: {self.epochs}\n')
            f.write(f'Learning rate: {self.learning_rate}\n')
            f.write(f'Batch size: {self.batch_size}\n')
            f.write(f'Weight decay: {self.weight_decay}\n')
            f.write(f'Warmup ratio: {self.warmup_ratio}\n')

class AutoModel_manual(torch.nn.Module):
    def __init__(self, labels, super_labels, pretrained = 'bert-base-german-cased', super_loss_weight=0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained)
        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.classifier_fine = torch.nn.Linear(hidden_size, labels)
        self.classifier_super = torch.nn.Linear(hidden_size, super_labels)
        self.super_loss_weight = super_loss_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, super_labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        logits_fine = self.classifier_fine(pooled_output)
        logits_super = self.classifier_super(pooled_output)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits_fine, labels) + self.super_loss_weight * loss_fct(logits_super, super_labels)

        return SequenceClassifierOutput(loss=loss, logits=logits_fine)

class EarlyStoppingWandbLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Log the current epoch to wandb after each epoch
        wandb.log({"epoch": state.epoch})

    def on_train_end(self, args, state, control, **kwargs):
        # Log the total number of epochs completed at the end of training
        wandb.log({
            "epochs_completed": state.epoch,
            "early_stopped": state.epoch < args.num_train_epochs  # True if stopped early
        })

if __name__ == "__main__":
    m = ModelEdit()
    m.train_model()
    m.evaluate_model()