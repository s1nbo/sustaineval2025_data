

class LogSegmentScalingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs['model']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        scaling_info = ''
        if hasattr(model, 'segment1_scaling'):
            scaling_info = f'Scaling Target: {model.segment1_scaling.item():.4f}'
        
        accuracy_info = ''
        if logs is not None and 'eval_accuracy' in logs:
            log_accuracy = logs['eval_accuracy']
            accuracy_info = f'Eval Accuracy: {log_accuracy:.4f}'
        
        print(f'\n[{timestamp} - Step {state.global_step}] {scaling_info} {accuracy_info}')