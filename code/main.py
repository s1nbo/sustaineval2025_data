from model import Model
import wandb
import yaml
from ensamble_model import Model_Ensamble


'''
model.optuna_training(n_trials=100)
'''

if __name__ == "__main__":
    model = Model()
    model.optuna_training(n_trials=1000)

    '''
    e = Model_Ensamble()
    e.load_models('798', '878', '899', '837', '979')
    e.evaluate_ensamble_models()
    e.generate_ensamble_submission()
    '''