

class PrintLogger:
    def __init__(self, print_freq=10):
        self.i = 0

    def log(self, relaxation_score, feature_vector):
        self.i += 1
        if self.i == 10:
            print(f"relaxation_score = {relaxation_score} \nfeature_vector =\n {feature_vector}\n\n")
            self.i = 0

