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
    


class SingleLabel(SuperLabel):
    def __init__(self, super_label: int):
        SuperLabel.__init__(self)
        self.super_label = super_label
    
    def load_model(self, mode_path):
        # load correct model
        pass
    
    def split_data(self, self_class):
        # update the data to only get that data of correct super_class

        
        self.submission[self.submission['predicted_label'] == self.super_class]
    
    
    # for generate submission we can use ensamble = True and only take first value

        
    


    

def generate_super_class_submission(l1: SingleLabel, l2: SingleLabel, l3: SingleLabel, l4: SingleLabel):
    
    all_predictions = l1.submission + l2.submission + l3.submission + l4.submission





if __name__ == '__main__':
    model = SuperLabel()