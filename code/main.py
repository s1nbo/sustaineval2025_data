from model import Model
import wandb
import yaml
from ensamble_model import Model_Ensamble


if __name__ == "__main__":
    
    
    #model = Model()
    #model.optuna_training(n_trials=1000)

    
    e = Model_Ensamble()
    e.confidence = 1 # Does This change something?
    e.load_models('899', '878', '837','1999', '798', '1153g', '1872')
    e.evaluate_ensamble_models()
    e.generate_ensamble_submission()
  