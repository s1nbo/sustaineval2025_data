from model import Model
import wandb
import yaml

def sweep_entrypoint():
    model = Model()
    model.run_sweep_training(wandb.config)

if __name__ == "__main__":
    sweep_config = yaml.safe_load(open("sweep.yaml"))
    sweep_id = wandb.sweep(sweep_config, project="sustaineval")
    wandb.agent(sweep_id, function=sweep_entrypoint, count=50)





    print('EVALUATING AUTO MODEL')
    #model.evaluate_model()
    #model.generate_submission()

