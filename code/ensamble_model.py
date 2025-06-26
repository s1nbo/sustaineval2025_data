from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import Model
import os
from collections import defaultdict, Counter
class Model_Ensamble(Model):
    
    def load_models(self, *model_paths, confidence: bool = False, weight: int = 1):
        num_models = len(model_paths) + weight

        if num_models % 2 == 0:
            raise ValueError('Need odd number of models')
        
        self.models = model_paths
        print(f'Using Models: {self.models}')
        self.use_confidence = confidence
        self.weight = weight

    def evaluate_ensamble_models(self, top_level: bool = False, subclass = None):
        """
        Parameters:
            model_paths: Variable amount of directory names inside the results directory.
            Each directory should contain everything needed for a transformer model.
        """
        self.predictions = []
        for model in self.models:
            self.model_directory = os.path.join(self.result_path, model)
            pred = self.evaluate_model(ensamble=True) # returns dataframe with id, prediction and confidence
            self.predictions.append(pred) # list of dataframes
        
        if top_level:
            for i in range(self.weight):
                self.predictions.append(subclass[['id', 'predicted_label', 'confidence_score']])


        y_true =  self.validation['task_a_label']
        ensemble_preds, _ = self.ensamble_prediction(submission=False)
        print(f'Accuracy: {accuracy_score(y_true, ensemble_preds):.4f}')

        # Classification Report
        with open(os.path.join(self.result_path, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(classification_report(y_true=y_true, y_pred=ensemble_preds, target_names=[self.label_name[i] for i in range(len(self.label_name))]))

        # Confusion Matrix
        cm = confusion_matrix(y_true, ensemble_preds)

        plt.figure(figsize=(14,12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[self.label_name[i] for i in range(len(self.label_name))], yticklabels=[self.label_name[i] for i in range(len(self.label_name))])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'confusion_matrix.png'))
        
    def ensamble_prediction(self, submission: bool):
        """
        Computes the accuracy of the ensemble model using majority voting.
        """
        # TODO
        # Submission or validation
        dfs = self.ensamble_submission if submission else self.predictions

        # check for misisng data
        all_ids = []
        expected_count = len(dfs)

        for df in dfs: all_ids.extend(df['id'].tolist())
        
        id_counts = Counter(all_ids)
    
        invalid_ids = [id for id, count in id_counts.items() if count != expected_count]
        
        if invalid_ids:
            raise ValueError(f"The following IDs do not appear in all model outputs: {invalid_ids}")


        # Merge all dataframes on 'id'
        all_ids = dfs[0][['id']]
        for df in dfs[1:]:
            all_ids = all_ids.merge(df[['id']], on='id', how='inner')  # common ids

        common_ids = all_ids['id'].tolist()

        # Per sample prediction and confidence lists
        predictions_per_sample = []
        confidence_per_sample = []

        for sample_id in common_ids:
            preds = []
            confs = []

            for df in dfs:
                row = df[df['id'] == sample_id].iloc[0]  # assuming ids are unique
                preds.append(row['predicted_label'])

                # Using the confidence decreases accuracy
                if self.use_confidence:
                    confs.append(row['confidence_score'])
                else:
                    confs.append(1.0)

            predictions_per_sample.append(preds)
            confidence_per_sample.append(confs)


        ensemble_preds = []
        ensemble_confidences = []
        
        for preds, confs in zip(predictions_per_sample, confidence_per_sample):
            weighted_votes = defaultdict(float)

            # Accumulate weighted votes
            for pred, conf in zip(preds, confs):
                weighted_votes[pred] += conf

            # Select label with highest total weight (confidence sum)
            majority_label = max(weighted_votes.items(), key=lambda x: x[1])[0]
            #total_weight = weighted_votes[majority_label]

            ensemble_preds.append(majority_label)
            #ensemble_confidences.append(total_weight)

        return ensemble_preds, ensemble_confidences

         
    def generate_ensamble_submission(self, top_level: bool = False, subclass = None):
        self.ensamble_submission = []
        for model in self.models:
            self.model_directory = os.path.join(self.result_path, model)
            pred = self.generate_submission(ensamble=True) # returns dataframe with id, prediction and confidence
            self.ensamble_submission.append(pred) # list of dataframes
        
        if top_level:
            for i in range(self.weight):
                self.ensamble_submission.append(subclass[['id', 'predicted_label', 'confidence_score']])

        
        self.submission['predicted_label'], _ = self.ensamble_prediction(submission=True)

        # Save the predictions to a CSV filef
        with open(os.path.join(self.result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
            f.write('id,label\n')
            for prediction in self.submission.iterrows():
                f.write (f"{prediction[1]['id']},{prediction[1]['predicted_label']}\n")