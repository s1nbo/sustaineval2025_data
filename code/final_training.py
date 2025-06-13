from model import Model
import pandas as pd 
import os 

from datasets import Dataset as HFDataset
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer)



# Class that includes validation data set in model training for final submission
# Pro: More Data, Con: Can't check if model got worse or better

class FinalModel(Model):

    def load_data(self, target = 'evaluation', top_class: bool = False):
        # Load Data, we want to combine training and trial for training
        self.training = pd.DataFrame()
        self.submission = pd.DataFrame() if not top_class else None
        self.data_files = ['trial', 'training','development', 'generated', target]
        if top_class: self.data_files = self.data_files[:-1] 

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
                    current_file['task_a_label'] = current_file['task_a_label'].apply(lambda x : self.top_level_labels[x])

            if file_name == target:
                self.submission = current_file
            else:
                self.training = pd.concat([self.training, current_file], ignore_index=True)



    def train_model(self):
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
            data_collator=data_collator,
        )

        # Training
        trainer.train()
        print(f'Finished Training')

        # Save Model
        trainer.save_model(self.model_directory)
        self.tokenizer.save_pretrained(self.model_directory)