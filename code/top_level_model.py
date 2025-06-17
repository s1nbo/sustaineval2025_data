from model import Model

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
    
    def sort_data(self, super_label):
        # split the data based on the task_a_label in self.submission['predicted_label']
        # Only for submission needed, can be split correctly for the other classes
        return self.submission[self.submission['predicted_label'] == super_label]





class SingleLabel(SuperLabel):
    def __init__(self, super_class: int):
        SuperLabel.__init__(self)
        self.super_class = super_class


    def load_data(self):
        self.submission = self.sort_data(super_label=self.super_class)

    def generate_submission(self):
        pass

    # Maybe the rest can be used from Parent class?



if __name__ == '__main__':
    model = SuperLabel()
    model.train_model()
    model.evaluate_model()
    a, _, _, _ = model.sort_data()
    print(a)
