import numpy as np
import pandas as pd


class SVDRecommender_Test:

    def __init__(self, df_path: str='./data/svd-rec-ratings-df-test.csv',
        complete_df_path: str='./data/ratings.csv') ->None:
        self.ratings_df = pd.read_csv(df_path)
        self.complete_ratings_df = pd.read_csv(complete_df_path)
        self.test_user_id = 660
        self.movies_pool = np.array(['Ant-Man (2015)', 'Iron Man 2 (2010)',
            'Avengers: Age of Ultron (2015)', 'Thor (2011)',
            'Captain America: The First Avenger (2011)',
            'Man of Steel (2013)',
            'Star Wars: Episode IV - A New Hope (1977)',
            'Ladybird Ladybird (1994)', 'Man of the House (1995)',
            'Jungle Book, The (1994)'])
        self.predict_expected_outputs = [
            'Captain America: The First Avenger (2011)', 'Ant-Man (2015)',
            'Avengers: Age of Ultron (2015)']
        self.slice_U_k = np.array([[-0.288836854063865, -0.882180547612558,
            -1.894903678810819], [0.63987069625696, 0.62553014479506, 
            1.637035875683824], [1.239878749911288, 0.598290692949525, 
            0.987317344608135]])
        self.slice_V_k = np.array([[-0.322567934474643, -0.194226144368913,
            0.270272345422909], [0.050056569696058, 0.036580660973925, 
            0.313574856965127], [0.572301244929592, 1.318040758707676, -
            0.336799937534724]])
        self.U_k_expected_shape = 50, 10
        self.V_k_expected_shape = 10, 200

    def get_slice_UV(self, U_k: np.ndarray, V_k: np.ndarray) ->np.ndarray:
        """
        Gets the slice of U_k and V_k to compare with

        Args:
            U_k: np.ndarray
            V_k: np.ndarray
        """
        slice_U_k = U_k[20:23, 2:5]
        slice_V_k = V_k[2:5, 20:23]
        return slice_U_k, slice_V_k
