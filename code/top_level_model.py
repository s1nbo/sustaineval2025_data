from model import Model

class TopLevel(Model):
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


if __name__ == '__main__':
    t = TopLevel()
    m = Model()
