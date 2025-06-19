from model import Model
import os
import pandas as pd

'''
Without tuning the Models I've reached 0.64 Accuracy (0.81*0.79).
'''

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
        
    
    def split_data(self, super_label):
        # Splits data based on prediction
        self.validation = self.validation[self.validation['predicted_label'] == super_label]
        self.submission = self.submission[self.submission['predicted_label'] == super_label]

        return self.validation, self.submission
    
    # Use self.generate_submission(ensamble=True) to get submission data
    


class SingleLabel(Model):
    def __init__(self, super_label: int):
        Model.__init__(self)
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

        if self.super_label == 1: 
            self.training['task_a_label'] = self.training['task_a_label'] - 4
            self.validation['task_a_label'] = self.validation['task_a_label'] - 4

        if self.super_label == 2: 
            self.training['task_a_label'] = self.training['task_a_label'] - 10
            self.validation['task_a_label'] = self.validation['task_a_label'] - 10
        
        if self.super_label == 3: 
            self.training['task_a_label'] = self.training['task_a_label'] - 13
            self.validation['task_a_label'] = self.validation['task_a_label'] - 13
        
        

    def recover_original_label(self):
        if self.super_label == 1: 
            self.training['task_a_label'] = self.training['task_a_label'] + 4
            self.validation['task_a_label'] = self.validation['task_a_label'] + 4
            # self.validation['predicted_label'] = self.validation['predicted_label'] + 4 TODO

        if self.super_label == 2: 
            self.training['task_a_label'] = self.training['task_a_label'] + 10
            self.validation['task_a_label'] = self.validation['task_a_label'] + 10
            # self.validation['predicted_label'] = self.validation['predicted_label'] + 10 TODO
        
        if self.super_label == 3: 
            self.training['task_a_label'] = self.training['task_a_label'] + 13
            self.validation['task_a_label'] = self.validation['task_a_label'] + 13
            # self.validation['predicted_label'] = self.validation['predicted_label'] + 13 TODO

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
    
    # Tuning
    # SuperLabel: 0.81
    # Train and evaluate the super model
    #super_model = SuperLabel()
    #super_model.train_model()
    #super_model.evaluate_model()
    #super_model.generate_submission(ensamble=True) TODO

    # Store the submission results for each subclass
    subclass_submissions = []


    # Without Tuning
    # Label0: 0.76
    # Label1: 0.80
    # Label2: 0.71
    # Label3: 0.90
    # Average: 0.7925
    for super_class in range(4):
        model = SingleLabel(super_class)
        #model.train_model()
        # model.evaluate_model()
        model.recover_original_label()
        
        # TODO 
        # model.split_data(super_model.submission)
        # submission, _ = model.generate_submission(ensamble=True)
        # subclass_submissions.append(submission)

    
    #Generate final combined submission
    #generate_super_class_submission(*subclass_submissions, result_path=model.result_path)



    #