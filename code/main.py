from model import Model

model = Model()

print('TRAINING AUTO MODEL')
model.train_auto_model(test=False)
print('EVALUATING AUTO MODEL')
model.evaluate_model()
model.generate_submission()