from model import Model
import os
from collections import Counter
import numpy as np

class Model_Ensamble(Model):
    def evaluate_ensamble_models(self, *model_paths):
        """
        Parameters:
            model_paths: Variable amount of directory names inside the results directory.
            Each directory should contain everything needed for a transformer model.
        """
        num_models = len(model_paths)
        
        if num_models % 2 == 0:
            raise ValueError('Need odd number of models')

        self.predictions = []
        for model in model_paths:
            self.model_directory = os.path.join(self.result_path, model)
            self.predictions.append(self.evaluate_model(ensamble=True))
        
        print(self.predictions)
        print(type(self.predictions[0]))
                               
    def ensamble_accuracy(self):
        """
        Computes the accuracy of the ensemble model using majority voting.
        Assumes self.predictions is a list of lists (one list per model's predictions).
        Assumes self.true_labels contains the ground truth labels.
        """
        y_true =  self.validation['task_a_label']

        # Transpose to get predictions for each example across all models
        predictions_per_sample = list(zip(*self.predictions))

        ensemble_preds = []
        for preds in predictions_per_sample:
            vote = Counter(preds).most_common(1)[0][0]
            ensemble_preds.append(vote)

        ensemble_preds = np.array(ensemble_preds)
        true_labels = np.array(y_true)

        acc = np.mean(ensemble_preds == true_labels)
        print(f'Accuracy: {acc:.4f}')
        for i in range(len(ensemble_preds)):
            print(f'{ensemble_preds[i]} - {true_labels[i]}')
        

    # Maybe use 
    def hypertune_prediciton_weights(self):
        pass

    def generate_ensamble_submission(self):
        pass
        
'''
#     def generate_submission(self):
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
'''

e = Model_Ensamble()
e.evaluate_ensamble_models('1','2','3')
e.ensamble_accuracy()
