from model import Model
import wandb
import yaml
from ensamble_model import Model_Ensamble

if __name__ == "__main__":
    
    
    #model = Model()
    #model.optuna_training(n_trials=1000)

    
    e = Model_Ensamble()

    # ensamble = ['899', '878', '837','1999', '798', '1872', '2148', '815', '1153g']
    ensamble = ['899','1999', '798', '1872','878', '815', '1153g']
    e.load_models(*ensamble)
    e.evaluate_ensamble_models()
    # e.generate_ensamble_submission()
    
    # ['899','1999', '798', '1872','2148', '815', '1153g'] = 0.7341
    # ['899','1999', '798', '1872','878', '815', '1153g']