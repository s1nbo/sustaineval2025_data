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
    subclass_validation = pd.DataFrame()
    model_paths = ['label0', 'label1', 'label2', 'label3']


    for super_class in range(4):
        model = SingleLabel(super_label=super_class, path=model_paths[super_class])
        model.validation, model.submission = super_model.split_data(super_class)
        model.submission = model.submission.copy()
        model.validation = model.validation.copy()
        
        model.update_labels()
        model.evaluate_model(early_stop=True)
        model.generate_submission(early_stop=True)
        model.recover_original_label()
        
        submission = model.submission[['id', 'predicted_label' , 'confidence_score']]
        subclass_validation = pd.concat([subclass_validation, model.validation], ignore_index=True)
        subclass_submissions = pd.concat([subclass_submissions, submission], ignore_index=True)

    # for single submission
    # generate_super_class_submission(subclass_submissions, result_path=model.result_path)

    return subclass_validation, subclass_submissions


  
def ensamble(top_level: bool = False, vali = None, sub = None, confidence: bool = False, weight:int = 1):
    '''
    top_level: Should the the top_level_model be included
    weight: How should strongly the top_level_model is weighted (1x, 2x, ...) (if zero we ignore the top_level)
    vali = Validation from top_level
    sub = Submission from top_level
    confidence = Bool, should we use the confidence from the model between 0 and 1 or just use 1 for everything
    
    '''
    e = Model_Ensamble()

    # ensamble = ['899', '878', '837','1999', '798', '1872', '2148', '815', '1153g']
    model_names = ['899','1999', '798', '1872','878', '815', '1153g']
    e.load_models(*model_names, confidence=confidence, weight=weight)
    e.evaluate_ensamble_models(top_level=top_level, subclass=vali)
    e.generate_ensamble_submission(top_level=top_level, subclass= sub)
    
    # ['899','1999', '798', '1872','2148', '815', '1153g']
    # ['899','1999', '798', '1872','878', '815', '1153g'] = 0.7416, No Confidence


def training():
    model = Model()
    model.optuna_training(n_trials=1000)


def main():
    s_vali, s_sub = twolayers()
    ensamble(top_level=False, vali= s_vali,sub = s_sub, confidence=False, weight=2)


if __name__ == "__main__":
    main()

    