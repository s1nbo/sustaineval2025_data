from model import Model
import wandb
import yaml
import pandas as pd
from ensamble_model import Model_Ensamble
from top_level_model import SuperLabel, SingleLabel, generate_super_class_submission


def twolayers():
    super_model = SuperLabel('super')
    super_model.evaluate_model(super_label=True)
    super_model.generate_submission(super_label=True)
   
    # Store the submission results for each subclass
    subclass_submissions = pd.DataFrame()
    model_paths = ['label0', 'label1', 'label2', 'label3']


    for super_class in range(4):
        model = SingleLabel(super_label=super_class, path=model_paths[super_class])
        model.validation, model.submission = super_model.split_data(super_class)
        model.submission = model.submission.copy()
        model.validation = model.validation.copy()
        
        model.filter_validation()
        model.update_labels()
        model.evaluate_model(early_stop=True)
        model.generate_submission(early_stop=True)
        model.recover_original_label()
        
        submission = model.submission[['id', 'predicted_label' , 'confidence_score']]
        subclass_submissions = pd.concat([subclass_submissions, submission], ignore_index=True)
    
    generate_super_class_submission(subclass_submissions, result_path=model.result_path)

  
def ensamble():
    e = Model_Ensamble()

    # ensamble = ['899', '878', '837','1999', '798', '1872', '2148', '815', '1153g']
    ensamble = ['899','1999', '798', '1872','878', '815', '1153g']
    e.load_models(*ensamble)
    e.evaluate_ensamble_models()
    e.generate_ensamble_submission()
    
    # ['899','1999', '798', '1872','2148', '815', '1153g']
    # ['899','1999', '798', '1872','878', '815', '1153g'] = 0.7416, No Confidence


def training():
    model = Model()
    model.optuna_training(n_trials=1000)



if __name__ == "__main__":
    twolayers()
    