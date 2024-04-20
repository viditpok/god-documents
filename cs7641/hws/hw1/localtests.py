import pickle
import unittest
import uuid
import numpy as np
from warmup import *


class WarmupTests(unittest.TestCase):

    def setUp(self):
        self.arr1d = np.array([1, 3, 6, 2, 3, 5, 8, 4, 2, 1, 5, 7, 9, 3])
        self.arr2d = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    def test_get_packages(self):
        """
        Args:
            package_list: a python list of strings representing packages the runtime can see
        Return:
            None
        """
        req_packages = ['matplotlib', 'numpy', 'tqdm', 'imageio',
            'scikit-learn', 'notebook', 'statsmodels', 'jupyterlab',
            'scikit-image', 'pandas', 'ipywidgets', 'seaborn',
            'tweet-preprocessor']
        missing_packages = []
        with open('env.pkl', 'rb') as f:
            package_list = pickle.load(f)
        package_list = [str(p) for p in package_list]
        package_list = [p for p in package_list if not p.isdigit()]
        for package in req_packages:
            if not any(package in p for p in package_list):
                missing_packages.append(package)
        package_list.append(uuid.getnode())
        with open('env.pkl', 'wb') as f:
            pickle.dump(package_list, f)
        self.assertTrue(len(missing_packages) == 0, msg=
            f'Missing: {missing_packages}')
        print('Passed test_get_packages')

    def test_indices_of_k(self):
        student = indices_of_k(self.arr1d, 3)
        solution = np.array([1, 4, 13])
        self.assertTrue((solution == student).all())
        print('Correct Values for indices_of_k')

    def test_argmax_1d(self):
        student = argmax_1d(self.arr1d)
        solution = 12
        self.assertTrue(solution == student)
        print('Correct Values for argmax_1d')

    def test_mean_rows(self):
        student = mean_rows(self.arr2d)
        solution = np.array([1, 2, 3])
        self.assertTrue((solution == student).all())
        print('Correct Values for mean_rows')

    def test_sum_squares(self):
        student = sum_squares(self.arr2d)
        solution = np.array([[3], [12], [27]])
        if student.shape != solution.shape:
            print(
                f"""sum_squares output shape: 
 expected: {solution.shape}
 actual: {student.shape}"""
                )
        self.assertTrue((solution == student).all())
        print('Correct Values for sum_squares')


if __name__ == '__main__':
    unittest.main()
