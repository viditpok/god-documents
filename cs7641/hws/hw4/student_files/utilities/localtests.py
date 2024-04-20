import pickle
import unittest

import numpy as np
from NN import NeuralNet as dlnet
from random_forest import RandomForest
from utilities.utils import get_housing_dataset


class TestNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.setUp()

        # sample training data
        self.x_train = np.array(
            [
                [
                    0.72176308,
                    0.43961601,
                    0.13553666,
                    0.55713544,
                    0.87702101,
                    0.12019972,
                    0.04842653,
                    0.01573553,
                ],
                [
                    0.53498397,
                    0.81056978,
                    0.17524362,
                    0.00521916,
                    0.2053607,
                    0.90502607,
                    0.99638276,
                    0.45936163,
                ],
                [
                    0.84114195,
                    0.78107371,
                    0.62526833,
                    0.18139081,
                    0.28554493,
                    0.86342263,
                    0.11350829,
                    0.82592072,
                ],
                [
                    0.43286995,
                    0.13815595,
                    0.71456809,
                    0.985452,
                    0.60177364,
                    0.87152055,
                    0.85442663,
                    0.7442592,
                ],
                [
                    0.54714474,
                    0.45039175,
                    0.43588923,
                    0.53943311,
                    0.70734352,
                    0.67388256,
                    0.29136773,
                    0.19560766,
                ],
                [
                    0.5617591,
                    0.86315884,
                    0.34730499,
                    0.13892525,
                    0.53279486,
                    0.79825459,
                    0.37465092,
                    0.23443029,
                ],
                [
                    0.4233198,
                    0.0020612,
                    0.4777035,
                    0.78088463,
                    0.8208675,
                    0.76655747,
                    0.72102559,
                    0.79251294,
                ],
                [
                    0.74503529,
                    0.25137268,
                    0.76440309,
                    0.5790357,
                    0.03791042,
                    0.82510481,
                    0.64463256,
                    0.08997057,
                ],
                [
                    0.81644094,
                    0.51437913,
                    0.75881908,
                    0.96191336,
                    0.56525617,
                    0.70372399,
                    0.75134392,
                    0.56722149,
                ],
            ]
        )
        self.y_train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]).T

    def setUp(self):
        self.nn = dlnet(y=np.random.randn(1, 30), use_dropout=False, use_momentum=False)

    def assertAllClose(self, student, truth, msg=None):
        self.assertTrue(np.allclose(student, truth), msg=msg)

    def assertDictAllClose(self, student, truth):
        for key in truth:
            if key not in student:
                self.fail("Key " + key + " missing.")
            self.assertAllClose(student[key], truth[key], msg=(key + " is incorrect."))

        for key in student:
            if key not in truth:
                self.fail("Extra key " + key + ".")

    def test_leaky_relu(self):
        alpha = 0.05
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        u_copy = u.copy()
        student = self.nn.leaky_relu(alpha, u)
        truth = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.04886389,
                ],
                [
                    0.95008842,
                    -0.00756786,
                    -0.00516094,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.01025791,
                ],
                [
                    0.3130677,
                    -0.04270479,
                    -0.12764949,
                    0.6536186,
                    0.8644362,
                    -0.03710825,
                ],
                [
                    2.26975462,
                    -0.07271828,
                    0.04575852,
                    -0.00935919,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        self.assertAllClose(u, u_copy)
        print_success_message("test_leaky_relu")

    def test_softmax(self):
        input = np.array([[2, 0, 1], [1, 0, 2]])

        actual = self.nn.softmax(input)

        expected = np.array(
            [[0.66524096, 0.09003057, 0.24472847], [0.24472847, 0.09003057, 0.66524096]]
        )

        assert np.allclose(actual, expected, atol=0.1)
        print_success_message("test_softmax")

    def test_d_leaky_relu(self):
        alpha = 0.05
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_leaky_relu(alpha, u)
        truth = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.05],
                [1.0, 0.05, 0.05, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.05],
                [1.0, 0.05, 0.05, 1.0, 1.0, 0.05],
                [1.0, 0.05, 1.0, 0.05, 1.0, 1.0],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_leaky_relu")

    def test_dropout(self):
        np.random.seed(0)
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student, _ = self.nn._dropout(u, prob=0.3)

        truth = np.array(
            [
                [2.52007479, 0.57165316, 1.39819711, 3.201276, 2.66793999, -1.39611126],
                [
                    1.35726917,
                    -0.21622459,
                    -0.1474555,
                    0.58656929,
                    0.20577653,
                    2.07753359,
                ],
                [1.08719676, 0.17382146, 0.0, 0.0, 0.0, -0.29308323],
                [
                    0.44723957,
                    -1.22013677,
                    -3.64712831,
                    0.93374086,
                    1.23490886,
                    -1.06023574,
                ],
                [0.0, -2.07766524, 0.0, -0.2674055, 2.18968459, 2.09908396],
            ]
        )

        self.assertAllClose(student, truth)
        print_success_message("test_dropout")

    def test_loss(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])

        # Model's predicted probabilities for each class
        yh = np.array(
            [
                [0.8, 0.15, 0.05],
                [0.1, 0.7, 0.2],
                [0.05, 0.1, 0.85],
                [0.9, 0.05, 0.05],
                [0.1, 0.3, 0.6],
            ]
        )

        # Calculate Cross-Entropy
        student = self.nn.cross_entropy_loss(y, yh)

        truth = 0.2717047128349055

        self.assertAllClose(student, truth)
        print_success_message("test_loss")

    def test_forward_without_dropout(self):
        # load nn parameters
        file = open("data/test_data/nn_param.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(file))
        file.close()

        np.random.seed(42)

        x = np.random.rand(10, 3)

        student = self.nn.forward(x, use_dropout=False)

        truth = np.ones((10, 1))

        self.assertAllClose(student, truth)
        print_success_message("test_forward_without_dropout")

    def test_forward(self):
        # control random seed
        np.random.seed(0)

        # load nn parameters
        file = open("data/test_data/nn_param.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(file))
        file.close()

        np.random.seed(42)

        x = np.random.rand(10, 3)

        student = self.nn.forward(x, use_dropout=True)

        truth = np.ones((10, 1))

        self.assertAllClose(student, truth)
        print_success_message("test_forward")

    def test_compute_gradients_without_dropout(self):
        nn_param = open("data/test_data/nn_param.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        cache = open("data/test_data/test_compute_gradients_cache.pickle", "rb")
        self.nn.cache = pickle.load(cache)
        cache.close()

        for p in self.nn.cache:
            self.nn.cache[p] = self.nn.cache[p].T

        y = np.array(
            [
                [
                    0.77741921,
                    -0.11877117,
                    -0.19899818,
                    1.86647138,
                    -0.4189379,
                    -0.47918492,
                    -1.95210529,
                    -1.40232915,
                    0.45112294,
                    -0.6949209,
                    0.5154138,
                    -1.11487105,
                    -0.76730983,
                    0.67457071,
                    1.46089238,
                    0.5924728,
                    1.19783084,
                    1.70459417,
                    1.04008915,
                    -0.91844004,
                    -0.10534471,
                    0.63019567,
                    -0.4148469,
                    0.45194604,
                    -1.57915629,
                    -0.82862798,
                    0.52887975,
                    -2.23708651,
                    -1.1077125,
                    -0.01771832,
                ]
            ]
        )
        yh = np.array(
            [
                [
                    -1.71939447,
                    0.057121,
                    -0.79954749,
                    -0.2915946,
                    -0.25898285,
                    0.1892932,
                    -0.56378873,
                    0.08968641,
                    -0.6011568,
                    0.55607351,
                    1.69380911,
                    0.19686978,
                    0.16986926,
                    -1.16400797,
                    0.69336623,
                    -0.75806733,
                    -0.8088472,
                    0.55743945,
                    0.18103874,
                    1.10717545,
                    1.44287693,
                    -0.53968156,
                    0.12837699,
                    1.76041518,
                    0.96653925,
                    0.71304905,
                    1.30620607,
                    -0.60460297,
                    0.63658341,
                    1.40925339,
                ]
            ]
        )

        y = y.T
        yh = yh.T

        # print(yh.shape)
        self.nn.cache["o2"] = np.copy(yh)
        student = self.nn.compute_gradients(y, yh, use_dropout=False)

        truth = {
            "theta1": np.array(
                [
                    [-0.07785091, -0.01535662, -0.06638261, 0.00452876, -0.00989115],
                    [0.06728782, -0.00192398, 0.05364855, 0.00106911, -0.04182411],
                    [0.04498234, 0.0061795, -0.02079071, -0.00161607, 0.01896572],
                ]
            ),
            "b1": np.array(
                [-0.18226, -0.00617173, -0.09089671, 0.00203928, 0.03729976]
            ),
            "theta2": np.array(
                [[0.27774789], [0.21578218], [0.10773953], [0.15767398], [0.08500808]]
            ),
            "b2": np.array([0.27366111]),
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_compute_gradients_without_dropout")

    def test_compute_gradients(self):
        nn_param = open("data/test_data/nn_param.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        cache = open("data/test_data/test_compute_gradients_cache.pickle", "rb")
        self.nn.cache = pickle.load(cache)
        cache.close()

        for p in self.nn.cache:
            self.nn.cache[p] = self.nn.cache[p].T

        y = np.array(
            [
                [
                    0.77741921,
                    -0.11877117,
                    -0.19899818,
                    1.86647138,
                    -0.4189379,
                    -0.47918492,
                    -1.95210529,
                    -1.40232915,
                    0.45112294,
                    -0.6949209,
                    0.5154138,
                    -1.11487105,
                    -0.76730983,
                    0.67457071,
                    1.46089238,
                    0.5924728,
                    1.19783084,
                    1.70459417,
                    1.04008915,
                    -0.91844004,
                    -0.10534471,
                    0.63019567,
                    -0.4148469,
                    0.45194604,
                    -1.57915629,
                    -0.82862798,
                    0.52887975,
                    -2.23708651,
                    -1.1077125,
                    -0.01771832,
                ]
            ]
        )
        yh = np.array(
            [
                [
                    -1.71939447,
                    0.057121,
                    -0.79954749,
                    -0.2915946,
                    -0.25898285,
                    0.1892932,
                    -0.56378873,
                    0.08968641,
                    -0.6011568,
                    0.55607351,
                    1.69380911,
                    0.19686978,
                    0.16986926,
                    -1.16400797,
                    0.69336623,
                    -0.75806733,
                    -0.8088472,
                    0.55743945,
                    0.18103874,
                    1.10717545,
                    1.44287693,
                    -0.53968156,
                    0.12837699,
                    1.76041518,
                    0.96653925,
                    0.71304905,
                    1.30620607,
                    -0.60460297,
                    0.63658341,
                    1.40925339,
                ]
            ]
        )

        y = y.T
        yh = yh.T
        self.nn.cache["o2"] = np.copy(yh)
        student = self.nn.compute_gradients(y, yh, use_dropout=True)

        truth = {
            "theta1": np.array(
                [
                    [-0.10146069, 0.0145839, -0.04601286, 0.00931069, -0.02466569],
                    [0.10639195, -0.01048318, -0.01250874, 0.00034275, -0.0718992],
                    [0.01276157, 0.00178145, -0.007834, -0.00338637, 0.00247914],
                ]
            ),
            "b1": np.array(
                [-0.18736648, -0.02016031, -0.08800997, 0.00115881, 0.02780856]
            ),
            "theta2": np.array(
                [[0.27774789], [0.21578218], [0.10773953], [0.15767398], [0.08500808]]
            ),
            "b2": np.array([0.27366111]),
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_compute_gradients")

    def test_update_weights(self):
        nn_param = open("data/test_data/nn_param.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        dLoss_file = open("data/test_data/dLoss.pickle", "rb")
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()

        for p in dLoss:
            dLoss[p] = dLoss[p].T

        self.nn.update_weights(dLoss, use_momentum=False)
        student = self.nn.parameters

        theta1 = np.array(
            [
                [0.94882242, -0.36464497, -0.31395601],
                [-0.62450367, 0.49063477, -1.32195671],
                [1.00859644, -0.43012531, 0.18687619],
                [-0.14927761, 0.85106502, -1.18545526],
                [-0.17927593, -0.21328183, 0.66129455],
            ]
        ).T
        b1 = np.array(
            [[0.00012665], [0.0111731], [-0.00234416], [-0.01659802], [-0.00742044]]
        ).T
        theta2 = np.array(
            [[-0.48996797, -0.06823595, -0.38511864, 0.00195402, 0.26013481]]
        ).T
        b2 = np.array([[0.00636996]]).T

        truth = {"theta1": theta1, "b1": b1, "theta2": theta2, "b2": b2}

        self.assertDictAllClose(student, truth)
        print_success_message("test_update_weights")

    def test_update_weights_with_momentum(self):
        nn_param = open("data/test_data/nn_param.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        change_file = open("data/test_data/nn_change.pickle", "rb")
        self.nn.change = pickle.load(change_file)
        change_file.close()

        for p in self.nn.change:
            self.nn.change[p] = self.nn.change[p].T

        dLoss_file = open("data/test_data/dLoss.pickle", "rb")
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()

        for p in dLoss:
            dLoss[p] = dLoss[p].T

        self.nn.update_weights(dLoss, use_momentum=True)
        student = self.nn.parameters

        theta1 = np.array(
            [
                [0.95811233, -0.37082579, -0.32209426],
                [-0.62619372, 0.49663111, -1.32627343],
                [1.00950105, -0.42710571, 0.19302648],
                [-0.1520303, 0.84710098, -1.1823376],
                [-0.18187881, -0.20756012, 0.65728525],
            ]
        ).T
        b1 = np.array(
            [[-0.00010619], [0.01210595], [-0.00183543], [-0.02094245], [-0.0111725]]
        ).T
        theta2 = np.array(
            [[-0.4926153, -0.06892446, -0.38550774, -0.00113789, 0.25897234]]
        ).T
        b2 = np.array([[0.0029572]]).T
        truth = {"theta1": theta1, "b1": b1, "theta2": theta2, "b2": b2}

        self.assertDictAllClose(student, truth)
        print_success_message("test_update_weights_with_momentum")

    def test_gradient_descent(self):
        x_train, y_train, x_test, y_test = get_housing_dataset()

        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_momentum=False)

        nn.gradient_descent(x_train, y_train, iter=3, local_test=True)

        gd_loss = np.array([1.18213491, 1.18013266, 1.17818406])
        gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-1)
        print("\nYour GD losses works within the expected range:", gd_loss_test)

    def test_batch_gradient_descent(self):
        x_train, y_train, x_test, y_test = get_housing_dataset()

        np.random.seed(0)
        nn = dlnet(
            y_train, lr=0.01, batch_size=6, use_dropout=False, use_momentum=False
        )

        bgd_loss = np.array([1.106816, 1.112495, 1.301159])

        batch_y = np.array(
            [
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
            ]
        )
        batch_y = batch_y.reshape((3, 6, 3))

        nn.batch_gradient_descent(
            x_train, y_train, iter=3, local_test=True, use_momentum=False
        )
        batch_str = "batch_y at iteration %i: "
        print("\ny_train input:", y_train)
        [print(batch_str % (i), batch_y) for i, batch_y in enumerate(nn.batch_y)]

        bgd_loss_test = np.allclose(np.array(nn.loss), bgd_loss, rtol=1e-1)
        print("\nYour BGD losses works within the expected range:", bgd_loss_test)

        batch_y_test = np.allclose(np.array(nn.batch_y), batch_y, rtol=1e-1)
        print("Your batch_y works within the expected range:", batch_y_test)

    def test_gradient_descent_with_momentum(self):
        gd_loss_with_momentum = [1.182135, 1.180133, 1.177207]
        np.random.seed(0)
        x_train, y_train, x_test, y_test = get_housing_dataset()
        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_momentum=True)
        nn.gradient_descent(
            x_train, y_train, iter=3, use_momentum=True, local_test=True
        )
        gd_loss_test_with_momentum = np.allclose(
            np.array(nn.loss), gd_loss_with_momentum, rtol=1e-2
        )
        print(
            "\nYour GD losses works within the expected range:",
            gd_loss_test_with_momentum,
        )


class TestRandomForest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_bootstrapping(self):
        test_seed = 1
        num_feats = 40
        max_feats = 0.65
        rf_test = RandomForest(4, 5, max_feats)

        row_idx, col_idx = rf_test._bootstrapping(15, num_feats, test_seed)
        assert np.array_equal(
            row_idx, np.array([5, 11, 12, 8, 9, 11, 5, 0, 0, 1, 12, 7, 13, 12, 6])
        )
        assert np.array_equal(
            col_idx,
            np.array(
                [
                    30,
                    2,
                    16,
                    32,
                    31,
                    5,
                    34,
                    6,
                    15,
                    19,
                    10,
                    3,
                    21,
                    8,
                    39,
                    12,
                    24,
                    1,
                    7,
                    35,
                    26,
                    13,
                    22,
                    0,
                    27,
                    17,
                ]
            ),
        )
        print_success_message("test_bootstrapping")


def print_array(array):
    print(np.array2string(array, separator=", "))


def print_dict_arrays(dict_arrays):
    for key in dict_arrays:
        print(key)
        print_array(dict_arrays[key])


def print_success_message(test_name):
    print(test_name + " passed!")
