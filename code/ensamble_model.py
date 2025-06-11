from model import Model

class Model_Ensamble(Model):

    def __init__(self):
        pass
    
    def evaluate_ensamble_models(self, *model_paths):
        num_models = len(model_paths)
        
        if num_models % 2 == 0:
            raise ValueError('Need odd number of models')

        # pre process model path

        self.predictions = []
        for model in model_paths:
            self.model_directory = model 
            self.predictions.append(self.evaluate_model(ensamble=True))
                               

    def ensamble_accuracy(self):
        pass

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
e.evaluate_model(1, 2, 3)
