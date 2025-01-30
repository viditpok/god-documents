class FoodConsumptionPCA:
    def __init__(self, input_path = "data/food-consumption.csv"):
        '''
        LOAD the data
        '''
        raise NotImplementedError("Not Implemented")
        
    def country_pca(self):
        '''
        Returns (m, 2) numpy array for the first 2 principal components with food as feature vector.
        '''
        raise NotImplementedError("Not Implemented")

    def food_pca(self, num_dim = 2):
        '''
        Returns (m, 2) numpy array the first 2 principal components with country consumptions as feature vector.
        '''
        raise NotImplementedError("Not Implemented")
