from model import Model
import os
import pandas as pd



class SuperLabel(Model):
    def __init__(self, path:str = None):
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
        if path: self.load_model(model=path)
    
    def load_model(self, model:str):
        self.model_directory = os.path.join(self.result_path, model)

    def split_data(self, super_label:int):
        '''
        Before calling the functions
        self.evaluate_model(super_label=True) and 
        super_model.generate_submission(super_label=True)
        have to be called first
        '''
        super_label_vali = self.validation[self.validation['super_prediction'] == super_label]
        super_label_sub = self.submission[self.submission['super_prediction'] == super_label]

        return super_label_vali, super_label_sub


class SingleLabel(Model):
    def __init__(self, super_label: int, path:str = None):
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
        
        self.super_label_map = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7, 8, 9],
            2: [10, 11, 12],
            3: [13, 14, 15, 16, 17, 18, 19]
        } 

        valid_labels = self.super_label_map.get(self.super_label)
        self.training = self.training[self.training['task_a_label'].isin(valid_labels)]

        if path: self.load_model(model=path)
    
    def load_model(self, model:str):
        self.model_directory = os.path.join(self.result_path, model)

    def filter_validation(self):
        valid_labels = self.super_label_map.get(self.super_label)
        self.validation = self.validation[self.validation['task_a_label'].isin(valid_labels)]

    def update_labels(self):
        if self.super_label == 1: 
            self.training['task_a_label'] -= 4
            self.validation['task_a_label'] -= 4
        if self.super_label == 2: 
            self.training['task_a_label'] -= 10 
            self.validation['task_a_label'] -= 10
        if self.super_label == 3: 
            self.training['task_a_label'] -= 13
            self.validation['task_a_label'] -= 13
    
    def recover_original_label(self):
        if self.super_label == 1: 
            self.training['task_a_label'] += 4
            self.validation['task_a_label'] += 4
            self.validation['predicted_label'] += 4
            self.submission['predicted_label'] += 4
        if self.super_label == 2: 
            self.training['task_a_label'] += 10 
            self.validation['task_a_label'] += 10
            self.validation['predicted_label'] += 10
            self.submission['predicted_label'] += 10 
        if self.super_label == 3: 
            self.training['task_a_label'] += 13
            self.validation['task_a_label'] += 13
            self.validation['predicted_label'] += 13
            self.submission['predicted_label'] += 13
        
    # for generate submission we can use ensamble = True and only take first value

        

def generate_super_class_submission(submission, result_path):  
    
    with open(os.path.join(result_path, 'prediction_task_a.csv'), 'w', encoding='utf-8') as f:
        f.write('id,label\n')
        for _, prediction in submission.iterrows():
            f.write (f"{prediction['id']},{prediction['predicted_label']}\n")


#With  I've reached 0.72 Accuracy (0.8127*0.88625).
# Label0: 0.875
# Label1: 0.90
# Label2: 0.87
# Label3: 0.90
# Average: 0.88625

if __name__ == '__main__':
    super_model = SuperLabel()
    super_model.load_model('super')
    super_model.evaluate_model(super_label=True)
    super_model.generate_submission(super_label=True)


    # Train 
    model = SingleLabel(0)
    model.validation, model.submission = super_model.split_data(0)
    model.submission = model.submission.copy()
    model.validation = model.validation.copy()
    model.filter_validation()
    model.update_labels()
    model.train_model()

    # Evaluate
    model = SingleLabel(0)
    model.validation, model.submission = super_model.split_data(0)
    model.submission = model.submission.copy()
    model.validation = model.validation.copy()
    model.update_labels()
    model.evaluate_model(early_stop=True)
    model.recover_original_label()