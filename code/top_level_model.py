from model import Model
import os
import pandas as pd
class SuperLabel(Model):
    def __init__(self):
        self.setup()

        self.label_name = ['Strategy', 'Process Management', 'Environment', 'Society']
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

        self.load_data(top_class=True)
    

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
    

class SingleLabel(Model):
    def __init__(self, super_label: int):
        SuperLabel.__init__(self)
        self.super_label = super_label

        # Update labels to only get the necessary ones
        if super_label == 0:
            self.label_name = {
            0: 'Strategic Analysis and Action',   1: 'Materiality',
            2: 'Objectives',                      3: 'Depth of the Value Chain'
            }
        elif super_label == 1:
            self.label_name = {
            4: 'Responsibility',                  5: 'Rules and Processes',
            6: 'Control',                         7: 'Incentive Systems',
            8: 'Stakeholder Engagement',          9: 'Innovation and Product Management'
            }    
        elif super_label == 2:
            self.label_name = {
            10: 'Usage of Natural Resources',     11: 'Resource Management',
            12: 'Climate-Relevant Emissions'
            }
        else:
            self.label_name = {
            13: 'Employment Rights',
            14: 'Equal Opportunities',            15: 'Qualifications',
            16: 'Human Rights',                   17: 'Corporate Citizenship',
            18: 'Political Influence',            19: 'Conduct that Complies with the Law and Policy'
            }
        
        super_label_map = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7, 8, 9],
            2: [10, 11, 12],
            3: [13, 14, 15, 16, 17, 18, 19]
        } 

        valid_labels = super_label_map[self.super_label]
        self.training = self.training[self.training['task_a_label'].isin(valid_labels)]
        self.validation = self.validation[self.validation['task_a_label'].isin(valid_labels)]

    def split_data(self, submission):
            self.submission =  submission[submission['predicted_label'] == self.super_label]
        
    # Train can be taken from Parent
    
    # Do I really need this?
    def eval_single_model(self, model_path):
        self.model_directory = os.path.join(self.result_path, model_path)
        pred, conf = self.evaluate_model(ensamble=True)
        return pred
        
    # for generate submission we can use ensamble = True and only take first value

    # optuna_training can also be taken from Parent 
        

def generate_super_class_submission(*submissions, result_path):
    if len(submissions) != 4:
        raise ValueError(f"Expected 4 submissions, but got {len(submissions)}.")
    
    all_predictions = pd.concat(submissions, ignore_index=True)
    
    with open(os.path.join(result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
        f.write('id,label\n')
        for _, prediction in all_predictions.iterrows():
            f.write (f"{prediction['id']},{prediction['predicted_label']}\n")



if __name__ == '__main__':
    # Train and evaluate the super model
    super_model = SuperLabel()
    super_model.train_model()
    super_model.evaluate_model()
    #super_model.generate_submission(ensamble=True) TODO

    # Store the submission results for each subclass
    subclass_submissions = []

    # Loop through all super class indices (0–3)
    for super_class in range(4):
        model = SingleLabel(super_class)
        model.train_model()
        model.evaluate_model()
        
        #TODO 
        #model.split_data(super_model.submission)
        #submission, _ = model.generate_submission(ensamble=True)
        #subclass_submissions.append(submission)

    # Generate final combined submission
    #generate_super_class_submission(*subclass_submissions, result_path=model.result_path)
