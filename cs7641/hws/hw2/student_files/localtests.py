import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from dbscan import *
from gmm import *
from kmeans import *
from semisupervised import *
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.pairwise import euclidean_distances as sk_distance


def print_success_message():
    print("UnitTest passed successfully!")


class KMeansTests(unittest.TestCase):
    def runTest(self):
        pass

    def test_pairwise_dist(self, pwd=pairwise_dist):
        x = np.random.randn(2, 2)
        y = np.random.randn(3, 2)
        d1 = pairwise_dist(x, y)
        d2 = sk_distance(x, y)
        self.assertTrue(
            d1.shape == d2.shape,
            msg="Incorrect matrix shape. Expected: %s got: %s" % (d1.shape, d2.shape),
        )
        self.assertTrue(
            np.allclose(d1, d2, atol=0.0001), msg="Incorrect distance values"
        )
        print_success_message()

    def test_pairwise_speed(self, pwd=pairwise_dist):
        x = np.random.randn(16384, 3)
        y = np.random.randn(16384, 3)
        tic = time.perf_counter()
        times_pairwise = []
        times_sklearn = []
        for _ in range(10):
            _ = pairwise_dist(x, y)
            t1 = time.perf_counter() - tic
            tic = time.perf_counter()
            _ = sk_distance(x, y)
            t2 = time.perf_counter() - tic
            times_pairwise.append(t1)
            times_sklearn.append(t2)
        t1 = np.mean(times_pairwise)
        t2 = np.mean(times_sklearn)
        ratio = t1 / t2
        self.assertTrue(
            ratio < 10,
            msg="Your implementation is >10x slower than sklearn. Did you use broadcasting?",
        )
        print_success_message()

    def test_kmeans_loss(self, km=KMeans):
        points = np.array(
            [
                [-1.43120297, 0.8786823],
                [-0.92082436, 0.48852312],
                [-0.98384412, -1.9580956],
                [0.16539071, -2.00230059],
                [1.13874072, -1.37183974],
                [-0.30292898, -1.90092874],
                [-1.96110084, 0.2529296],
                [0.8089156, -0.43945057],
                [-0.35643708, -0.16907999],
                [-1.15222464, 1.23224667],
            ]
        )
        cluster_idx = np.array([2, 0, 2, 2, 0, 1, 2, 2, 2, 0])
        centers = np.array(
            [
                [-0.29934073, 0.22471242],
                [-0.65337045, 0.31246784],
                [-0.24212177, -0.0697861],
            ]
        )
        kmeans = km(points, len(np.unique(cluster_idx)))
        kmeans.assignments = cluster_idx
        kmeans.centers = centers
        loss = kmeans.get_loss()
        self.assertTrue(
            np.isclose(loss, 26.490707917359302),
            msg="Expected: 26.490707917359302 got: %s" % loss,
        )
        print_success_message()

    def test_update_centers(self, km=KMeans):
        points = np.array(
            [
                [-1.43120297, 0.8786823],
                [-0.92082436, 0.48852312],
                [-0.98384412, -1.9580956],
                [0.16539071, -2.00230059],
                [1.13874072, -1.37183974],
                [-0.30292898, -1.90092874],
                [-1.96110084, 0.2529296],
                [0.8089156, -0.43945057],
                [-0.35643708, -0.16907999],
                [-1.15222464, 1.23224667],
            ]
        )
        cluster_idx = np.array([2, 0, 2, 2, 0, 1, 2, 2, 2, 0])
        old_centers = np.array(
            [
                [-0.29934073, 0.22471242],
                [-0.65337045, 0.31246784],
                [-0.24212177, -0.0697861],
            ]
        )
        kmeans = km(points, len(np.unique(cluster_idx)))
        kmeans.assignments = cluster_idx
        kmeans.centers = old_centers
        new_centers = kmeans.update_centers()
        expected_centers = [
            [-0.31143609, 0.11631002],
            [-0.30292898, -1.90092874],
            [-0.62637978, -0.57288581],
        ]
        self.assertTrue(
            np.allclose(new_centers, expected_centers, atol=0.0001),
            msg="Incorrect centers, check that means are computed correctly",
        )
        print_success_message()

    def test_init(self, km=KMeans):
        X, y_true = make_blobs(
            n_samples=4000, centers=4, cluster_std=0.7, random_state=0
        )
        X = X[:, ::-1]
        kmeans = km(X, 4)
        plt.figure(1)
        colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        for k, col in enumerate(colors):
            cluster_data = y_true == k
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)
        centers_init1 = kmeans.init_centers()
        try:
            centers_init2 = kmeans.kmpp_init()
            plt.scatter(centers_init1[:, 0], centers_init1[:, 1], c="k", s=50)
            plt.scatter(centers_init2[:, 0], centers_init2[:, 1], c="r", s=50)
            plt.legend(["1", "2", "3", "4", "random", "km++"])
        except NotImplementedError:
            plt.scatter(centers_init1[:, 0], centers_init1[:, 1], c="k", s=50)
            plt.legend(["1", "2", "3", "4", "random"])
        finally:
            plt.title("K-Means++ Initialization")
            plt.xticks([])
            plt.yticks([])
            plt.show()

    def test_rand_statistic(self):
        xPredicted = [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
        xGroundTruth = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
        r1 = rand_score(xGroundTruth, xPredicted)
        r2 = rand_statistic(xGroundTruth, xPredicted)
        print("Expected value: ", r1)
        print("Your value: ", r2, "\n")
        self.assertTrue(
            np.allclose(r1, r2, atol=0.0001), msg="Incorrect rand statistic calculation"
        )
        print_success_message()


class GMMTests(unittest.TestCase):
    def __init__(self) -> None:
        super().__init__()
        np.random.seed(5)
        self.data = np.random.randn(5, 4)
        self.points = np.random.randn(15, 3)
        self.mu = np.array(
            [
                [-0.69166075, -0.39675353, -0.6871727],
                [0.04221375, 0.58281521, -1.10061918],
                [1.62434536, -0.61175641, -0.52817175],
            ]
        )
        self.sigma = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        self.pi = np.ones(3) / 3

    def test_helper_functions(self):
        my_softmax = GMM(self.data, 3).softmax(self.data)
        expected_softmax = np.array(
            [
                [0.10782656, 0.04982049, 0.78844897, 0.05390398],
                [0.16080479, 0.70138883, 0.05805256, 0.07975382],
                [0.39637071, 0.23624673, 0.0996817, 0.26770086],
                [0.21741805, 0.56913777, 0.05890126, 0.15454293],
                [0.27040961, 0.54778227, 0.01886611, 0.16294201],
            ]
        )
        print(
            "Your softmax works within the expected range: ",
            np.allclose(expected_softmax, my_softmax),
        )
        my_logsumexp = GMM(self.data, 3).logsumexp(self.data)
        expected_logsumexp = np.array(
            [[2.66845878], [1.93717399], [1.11300859], [1.16710435], [2.4592084]]
        )
        print(
            "Your logsumexp works within the expected range: ",
            np.allclose(expected_logsumexp, my_logsumexp),
        )

    def test_init_components(self):
        points = self.points
        my_init_pi, my_init_mu, my_init_sigma = GMM(points, 3)._init_components()
        expected_init_pi = np.array([0.33333333, 0.33333333, 0.33333333])
        expected_init_sigma = np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ]
        )
        print(
            "Your _init_component's pi works within expected range: ",
            np.allclose(my_init_pi, expected_init_pi),
        )
        print(
            "Your _init_component's mu works within expected range: ",
            len(np.unique(my_init_mu, axis=0)) == 3,
        )
        print(
            "Your _init_component's sigma works within the expected range: ",
            np.allclose(my_init_sigma, expected_init_sigma),
        )

    def test_undergrad(self):
        points = self.points
        mu = self.mu
        pi = self.pi
        sigma = self.sigma
        data = self.data
        my_normalpdf = GMM(points, 3).normalPDF(points, mu[0], sigma[0])
        expected_normal_pdf = np.array(
            [
                0.05385474,
                0.00871279,
                0.03771877,
                0.02631223,
                0.039282,
                0.00442542,
                0.00722764,
                0.03184208,
                0.00181295,
                0.02640811,
                0.00471099,
                0.01150689,
                0.00063621,
                0.01147473,
                0.02380852,
            ]
        )
        print(
            "Your normal pdf works within the expected range: ",
            np.allclose(expected_normal_pdf, my_normalpdf),
        )
        my_lljoint = GMM(points, 3)._ll_joint(pi, mu, sigma, False)
        expected_lljoint = np.array(
            [
                [-4.02007719, -5.44099343, -7.3374222],
                [-5.84157502, -5.69251151, -8.01291517],
                [-4.37620973, -4.91245496, -5.25244293],
                [-4.73633379, -3.99850407, -6.34444602],
                [-4.33560118, -4.63992545, -5.42838328],
                [-6.51900277, -6.91677702, -6.86391589],
                [-6.02845446, -4.65853011, -7.70587119],
                [-4.5455788, -5.16685045, -6.70899907],
                [-7.4114131, -6.64144377, -7.97545579],
                [-4.73269633, -4.8801069, -4.6231013],
                [-6.45646958, -4.88294659, -4.53086302],
                [-5.56342186, -7.33396705, -7.94493792],
                [-8.45858816, -6.81718207, -10.18015438],
                [-5.56622065, -6.52577609, -6.12075969],
                [-4.83632403, -6.27755579, -6.21813739],
            ]
        )
        print(
            "Your lljoint works within the expected range: ",
            np.allclose(my_lljoint, expected_lljoint),
        )
        my_estep = GMM(points, 3)._E_step(pi, mu, sigma, False)
        expected_estep = np.array(
            [
                [0.78263086, 0.1889996, 0.02836954],
                [0.43960461, 0.5102696, 0.05012579],
                [0.49967803, 0.29228189, 0.20804008],
                [0.30379836, 0.63536137, 0.06084027],
                [0.482415, 0.35583974, 0.16174526],
                [0.4201512, 0.28226332, 0.29758548],
                [0.1952397, 0.76827858, 0.03648172],
                [0.60525648, 0.32518058, 0.06956295],
                [0.26819685, 0.57922475, 0.1525784],
                [0.33570952, 0.28969704, 0.37459344],
                [0.0788462, 0.38032345, 0.54083035],
                [0.79198478, 0.13482761, 0.07318761],
                [0.15769863, 0.81410708, 0.02819429],
                [0.51088176, 0.19569997, 0.29341827],
                [0.67215193, 0.15905541, 0.16879265],
            ]
        )
        print(
            "Your E step works within the expected range: ",
            np.allclose(my_estep, expected_estep),
        )
        my_pi, my_mu, my_sigma = GMM(points, 3)._M_step(expected_estep, False)
        expected_pi = np.array([0.43628293, 0.394094, 0.16962307])
        expected_mu = np.array(
            [
                [-0.22485851, -0.06104529, 0.33535978],
                [0.00642446, 0.72356306, 0.19284601],
                [0.36429099, 0.01967377, 0.11272403],
            ]
        )
        expected_sigma = np.array(
            [
                [
                    [0.20711668, 0.0, 0.0],
                    [0.0, 0.58982465, 0.0],
                    [0.0, 0.0, 0.51253382],
                ],
                [
                    [0.22612501, 0.0, 0.0],
                    [0.0, 0.88759053, 0.0],
                    [0.0, 0.0, 0.54600949],
                ],
                [[0.38012959, 0.0, 0.0], [0.0, 0.35395325, 0.0], [0.0, 0.0, 0.846592]],
            ]
        )
        print(
            "Your M step works within the expected range: ",
            np.allclose(my_pi, expected_pi)
            and np.allclose(my_mu, expected_mu)
            and np.allclose(my_sigma, expected_sigma),
        )

    def test_grad(self):
        points = self.points
        mu = self.mu
        pi = self.pi
        sigma = self.sigma
        data = self.data
        sigma_grad = np.array(
            [
                [
                    [0.30792796, 0.07909229, -0.11016917],
                    [0.07909229, 0.86422655, 0.06975468],
                    [-0.11016917, 0.06975468, 0.63212106],
                ],
                [
                    [0.30792796, 0.07909229, -0.11016917],
                    [0.07909229, 0.86422655, 0.06975468],
                    [-0.11016917, 0.06975468, 0.63212106],
                ],
                [
                    [0.30792796, 0.07909229, -0.11016917],
                    [0.07909229, 0.86422655, 0.06975468],
                    [-0.11016917, 0.06975468, 0.63212106],
                ],
            ]
        )
        my_multinormalpdf = GMM(data, 3).multinormalPDF(points, mu[0], sigma_grad[0])
        expected_multinormal_pdf = np.array(
            [
                0.12412461,
                0.0109187066,
                0.0283199962,
                0.0484688341,
                0.0396140713,
                0.00035509416,
                0.0115479475,
                0.0506551884,
                0.000337348839,
                0.00634219531,
                0.000120587696,
                0.00814168432,
                0.000576457373,
                0.00182882581,
                0.0149926366,
            ]
        )
        print(
            "Your multinormal pdf works within the expected range: ",
            np.allclose(expected_multinormal_pdf, my_multinormalpdf),
        )
        sigma_now = sigma * 0.5
        my_lljoint = GMM(points, 3)._ll_joint(pi, mu, sigma_now, True)
        expected_lljoint = np.array(
            [
                [-3.14500571, -5.9868382, -9.77969574],
                [-6.78800137, -6.48987436, -11.13068168],
                [-3.85727081, -4.92976126, -5.6097372],
                [-4.57751893, -3.10185949, -7.79374338],
                [-3.7760537, -4.38470224, -5.9616179],
                [-8.14285688, -8.93840538, -8.83268311],
                [-7.16176025, -4.42191156, -10.51659372],
                [-4.19600894, -5.43855224, -8.52284947],
                [-9.92767754, -8.38773887, -11.05576292],
                [-4.570244, -4.86506515, -4.35105393],
                [-8.01779049, -4.87074452, -4.16657737],
                [-6.23169506, -9.77278544, -10.99472719],
                [-12.02202766, -8.73921549, -15.46516011],
                [-6.23729264, -8.15640353, -7.34637072],
                [-4.77749941, -7.65996291, -7.54112612],
            ]
        )
        print(
            "Your lljoint works within the expected range: ",
            np.allclose(my_lljoint, expected_lljoint),
        )
        my_estep = GMM(points, 3)._E_step(pi, mu, sigma_now, True)
        expected_estep = np.array(
            [
                [0.94372325, 0.05503671, 0.00124004],
                [0.42366876, 0.57082286, 0.00550839],
                [0.65984771, 0.22577041, 0.11438188],
                [0.18470545, 0.80788672, 0.00740783],
                [0.60368247, 0.32845499, 0.06786254],
                [0.5120336, 0.23109797, 0.25686843],
                [0.06053431, 0.93735212, 0.00211357],
                [0.76813271, 0.22172086, 0.01014643],
                [0.16700188, 0.77894757, 0.05405055],
                [0.33447807, 0.24907403, 0.4164479],
                [0.01402184, 0.3262493, 0.65972887],
                [0.96383555, 0.0279336, 0.00823084],
                [0.0361238, 0.96272152, 0.00115468],
                [0.67723134, 0.09937515, 0.22339351],
                [0.8936077, 0.05003903, 0.05635326],
            ]
        )
        print(
            "Your E step works within the expected range: ",
            np.allclose(my_estep, expected_estep),
        )
        my_pi, my_mu, my_sigma = GMM(points, 3)._M_step(expected_estep, True)
        expected_pi = np.array([0.4828419, 0.39149886, 0.12565925])
        expected_mu = np.array(
            [
                [-0.26263543, -0.23026888, 0.37410807],
                [0.02946666, 0.96190945, 0.19697914],
                [0.64855787, -0.0282273, -0.12987796],
            ]
        )
        expected_sigma = np.array(
            [
                [
                    [0.18480413, 0.08971316, 0.08911991],
                    [0.08971316, 0.36907686, 0.08744919],
                    [0.08911991, 0.08744919, 0.48008412],
                ],
                [
                    [0.17533767, -0.0757907, -0.08561511],
                    [-0.0757907, 0.73833814, 0.16291358],
                    [-0.08561511, 0.16291358, 0.52631415],
                ],
                [
                    [0.35145756, 0.10609808, -0.51519293],
                    [0.10609808, 0.15893304, -0.08535478],
                    [-0.51519293, -0.08535478, 0.99893729],
                ],
            ]
        )
        print(
            "Your M step works within the expected range: ",
            np.allclose(my_pi, expected_pi)
            and np.allclose(my_mu, expected_mu)
            and np.allclose(my_sigma, expected_sigma),
        )


class DBScanTests(unittest.TestCase):
    def test_region_query(self):
        self.eps = 0.2
        self.minPoints = 3
        self.x = np.array(
            [
                [4.44921468, 0.99067444],
                [0.96302552, 1.79996854],
                [4.28865161, 1.07127892],
                [0.93103109, 2.09965384],
                [0.87714783, 2.20467543],
                [4.31595483, 1.05237385],
                [0.82344716, 2.14171114],
                [4.20605954, 1.16302588],
                [4.34484718, 0.96594819],
                [0.81225409, 2.08657429],
            ]
        )
        self.dbscan = DBSCAN(eps=self.eps, minPts=self.minPoints, dataset=self.x)
        neighborhood = self.dbscan.regionQuery(8)
        correct_neighborhood = [0, 2, 5, 8]
        self.assertFalse(
            set(neighborhood) - set(correct_neighborhood),
            msg="Expected %s as neighbors but got %s instead"
            % (correct_neighborhood, neighborhood),
        )
        print_success_message()

    def test_expand_cluster(self):
        self.eps = 0.2
        self.minPoints = 3
        self.x = np.array(
            [
                [4.44921468, 0.99067444],
                [0.96302552, 1.79996854],
                [4.28865161, 1.07127892],
                [0.93103109, 2.09965384],
                [0.87714783, 2.20467543],
                [4.31595483, 1.05237385],
                [0.82344716, 2.14171114],
                [4.20605954, 1.16302588],
                [4.34484718, 0.96594819],
                [0.81225409, 2.08657429],
            ]
        )
        self.dbscan = DBSCAN(eps=self.eps, minPts=self.minPoints, dataset=self.x)
        cluster_idx = np.zeros(len(self.x)) - 1
        visited_indices = {0}
        neighbor_indices = [0, 2, 5, 8]
        self.dbscan.expandCluster(0, neighbor_indices, 0, cluster_idx, visited_indices)
        correct_cluster_idx = [0.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0]
        correct_visited_indices = {0, 2, 5, 7, 8}
        self.assertTrue(
            np.array_equal(cluster_idx, correct_cluster_idx),
            msg="You should've visited {0, 2, 5, 7, 8}, did you catch'em all?",
        )
        print_success_message()


class SemisupervisedTests(unittest.TestCase):
    def test_data_separating_methods(self):
        data = np.array(
            [
                [1.0, 2.0, 3.0, 1],
                [1.0, np.nan, 3.0, 1],
                [7.0, np.nan, 9.0, 0],
                [7.0, 8.0, 9.0, 0],
                [26.0, 27.0, 28.0, np.nan],
                [2.0, 3.0, 4.0, np.nan],
                [16.0, 17.0, 18.0, 1],
                [np.nan, 17.0, 18.0, 1],
                [11.0, 12.0, 13.0, np.nan],
                [22.0, 23.0, 24.0, 0],
                [np.nan, 23.0, 24.0, 0],
                [19.0, 20.0, 21.0, np.nan],
            ]
        )
        complete_answer = np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [7.0, 8.0, 9.0, 0.0],
                [16.0, 17.0, 18.0, 1.0],
                [22.0, 23.0, 24.0, 0.0],
            ]
        )
        incomplete_answer = np.array(
            [
                [1.0, np.nan, 3.0, 1.0],
                [7.0, np.nan, 9.0, 0.0],
                [np.nan, 17.0, 18.0, 1.0],
                [np.nan, 23.0, 24.0, 0.0],
            ]
        )
        unlabeled_answer = np.array(
            [
                [26.0, 27.0, 28.0, np.nan],
                [2.0, 3.0, 4.0, np.nan],
                [11.0, 12.0, 13.0, np.nan],
                [19.0, 20.0, 21.0, np.nan],
            ]
        )
        my_complete = complete_(data)
        my_incomplete = incomplete_(data)
        my_unlabeled = unlabeled_(data)
        self.assertTrue(
            my_complete.shape == complete_answer.shape,
            msg="Expected %s as complete shape but got %s instead"
            % (complete_answer.shape, my_complete.shape),
        )
        self.assertTrue(
            np.all(np.isclose(my_complete, complete_answer, equal_nan=True)),
            msg="Incorrect complete_ method. Check for no NaN values",
        )
        self.assertTrue(
            my_incomplete.shape == incomplete_answer.shape,
            msg="Expected %s as incomplete shape but got %s instead"
            % (incomplete_answer.shape, my_incomplete.shape),
        )
        self.assertTrue(
            np.all(np.isclose(my_incomplete, incomplete_answer, equal_nan=True)),
            msg="Incorrect incomplete_ method. Check if only features have NaN values",
        )
        self.assertTrue(
            my_unlabeled.shape == unlabeled_answer.shape,
            msg="Expected %s as unlabeled shape but got %s instead"
            % (unlabeled_answer.shape, my_unlabeled.shape),
        )
        self.assertTrue(
            np.all(np.isclose(my_unlabeled, unlabeled_answer, equal_nan=True)),
            msg="Incorrect unlabeled_ method. Check if only lables have NaN values",
        )
        print_success_message()

    def test_cleandata(self):
        self.knn_cleaner = CleanData()
        complete_data = np.array(
            [
                [1.0, 2.0, 3.0, 1],
                [7.0, 8.0, 9.0, 0],
                [16.0, 17.0, 18.0, 1],
                [22.0, 23.0, 24.0, 0],
            ]
        )
        incomplete_data = np.array(
            [
                [1.0, np.nan, 3.0, 1],
                [7.0, np.nan, 9.0, 0],
                [np.nan, 17.0, 18.0, 1],
                [np.nan, 23.0, 24.0, 0],
            ]
        )
        correct_clean_data = np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [7.0, 8.0, 9.0, 0.0],
                [16.0, 17.0, 18.0, 1.0],
                [22.0, 23.0, 24.0, 0.0],
                [14.5, 23.0, 24.0, 0.0],
                [7.0, 15.5, 9.0, 0.0],
                [8.5, 17.0, 18.0, 1.0],
                [1.0, 9.5, 3.0, 1.0],
            ]
        )
        clean_data = self.knn_cleaner(incomplete_data, complete_data, 2)
        self.assertTrue(
            clean_data.shape == correct_clean_data.shape,
            msg="Expected %s as clean data shape but got %s instead"
            % (correct_clean_data.shape, clean_data.shape),
        )
        if np.all(np.isclose(clean_data, correct_clean_data)):
            print_success_message()
            return
        clean_data_sorted = np.array(sorted([tuple(row) for row in clean_data]))
        correct_clean_data_sorted = np.array(
            sorted([tuple(row) for row in correct_clean_data])
        )
        self.assertTrue(
            np.all(np.allclose(clean_data_sorted, correct_clean_data_sorted)),
            msg="Incorrect implementation. Check if all NaN values are replaced correctly",
        )
        print_success_message()

    def test_mean_clean_data(self):
        complete_data = np.array(
            [
                [1.0, 2.0, 3.0, 1],
                [7.0, 8.0, 15.0, 0],
                [16.0, 17.0, 18.0, 1],
                [22.0, 23.0, 24.0, 0],
                [1.0, np.nan, 3.0, 1],
                [7.0, 20.0, np.nan, 0],
                [np.nan, 17.0, 18.0, 1],
                [np.nan, 23.0, 24.0, 0],
            ]
        )
        correct_clean_data = np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [7.0, 8.0, 15.0, 0.0],
                [16.0, 17.0, 18.0, 1.0],
                [22.0, 23.0, 24.0, 0.0],
                [1, 15.7, 3, 1.0],
                [7.0, 20, 15.0, 0.0],
                [9.0, 17.0, 18.0, 1.0],
                [9.0, 23, 24.0, 0.0],
            ]
        )
        clean_data = mean_clean_data(complete_data)
        self.assertTrue(
            clean_data.shape == correct_clean_data.shape,
            msg="Expected %s as mean method clean data shape but got %s instead"
            % (correct_clean_data.shape, clean_data.shape),
        )
        if np.all(np.isclose(clean_data, correct_clean_data)):
            print_success_message()
            return
        clean_data_sorted = np.array(sorted([tuple(row) for row in clean_data]))
        correct_clean_data_sorted = np.array(
            sorted([tuple(row) for row in correct_clean_data])
        )
        self.assertTrue(
            np.all(np.allclose(clean_data_sorted, correct_clean_data_sorted)),
            msg="Incorrect implementation. Check if each feature's mean replaces all NaN values for the feature",
        )
        print_success_message()
