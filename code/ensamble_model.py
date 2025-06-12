from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import Model
import os
from collections import Counter

class Model_Ensamble(Model):
    
    def load_models(self, *model_paths):
        num_models = len(model_paths)

        if num_models % 2 == 0:
            raise ValueError('Need odd number of models')
        
        self.models = model_paths
        print(f'Using Models: {self.models}')

    def evaluate_ensamble_models(self):
        """
        Parameters:
            model_paths: Variable amount of directory names inside the results directory.
            Each directory should contain everything needed for a transformer model.
        """
        self.predictions = []
        for model in self.models:
            self.model_directory = os.path.join(self.result_path, model)
            self.predictions.append(self.evaluate_model(ensamble=True))

        y_true =  self.validation['task_a_label']
        ensemble_preds = self.ensamble_prediction(submission=False)
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
        # Transpose to get predictions for each example across all models
        if submission:
            predictions_per_sample = list(zip(*self.ensamble_submission))
        else:
            predictions_per_sample = list(zip(*self.predictions))

        ensemble_preds = []
        for preds in predictions_per_sample:
            vote = Counter(preds).most_common(1)[0][0]
            ensemble_preds.append(vote)
        
        return ensemble_preds
                         
    # TODO Maybe we want to weight models differently
    def hypertune_prediciton_weights(self):
        pass

    def generate_ensamble_submission(self):
        self.ensamble_submission = []
        for model in self.models:
            self.model_directory = os.path.join(self.result_path, model)
            self.ensamble_submission.append(self.generate_submission(ensamble=True))

        self.submission['predicted_label'] = self.ensamble_prediction(submission=True)

        # Save the predictions to a CSV file
        with open(os.path.join(self.result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
            f.write('id,label\n')
            for prediction in self.submission.iterrows():
                f.write (f"{prediction[1]['id']},{prediction[1]['predicted_label']}\n")



if __name__ == '__main__':
    e = Model_Ensamble()
    e.load_models('1', '2', '3')
    e.evaluate_ensamble_models()
    e.generate_ensamble_submission()



# TODO Maybe make everything a Dict so the values are confirmed ID bound 
