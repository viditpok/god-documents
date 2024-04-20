import numpy as np


class LogisticRegression_Test:

    def __init__(self) ->None:
        random_seed = 42
        N = 20
        D = 15
        self.lr = 0.1
        self.epochs = 10
        rng = np.random.default_rng(seed=random_seed)
        self.s = rng.random(size=(N, D))
        self.x = rng.random(size=(N, D))
        self.x_aug = rng.random(size=(N, D + 1))
        self.theta = rng.random(size=(D + 1, 1))
        self.h_x = rng.random(size=(N, 1))
        self.y = rng.integers(low=0, high=1, endpoint=True, size=(N, 1))
        self.y_hat = rng.integers(low=0, high=1, endpoint=True, size=(N, 1))
        self.sigmoid_result_slice = np.array([[0.684376044512838, 
            0.607991752965938, 0.702367636419746, 0.667603972237517], [
            0.556566477351084, 0.635198641021089, 0.515948901545992, 
            0.695853820750999], [0.678036329656022, 0.724622856476787, 
            0.580743276833795, 0.591570054945778], [0.69099269987157, 
            0.595675523290321, 0.57158677591335, 0.664295440255639]])
        self.bias_augment_slice_sum = 10.468940728428445
        self.predict_probs_result_slice = np.array([[0.992645546463048], [
            0.984174064480217], [0.97582404399237], [0.968212116486582]])
        self.predict_labels_result_slice = np.array([[1], [0], [1], [0]])
        self.loss_result = 1.3835952867282455
        self.gradient_result_slice = np.array([[0.020951220277763], [
            0.117433269774405], [0.078158311623874], [0.091212833534361]])
        self.accuracy_result = 0.5
        self.evaluate_result = 2.7672488636369814, 0.3
        self.fit_result_slice = np.array([[-0.122700330110435], [-
            0.096028185517518], [-0.047291068532804], [-0.052525438570197]])
