class OrderOfFaces:
    def __init__(self, images_path = 'data/isomap.mat'):
        '''
        Load the data from input files provided.
        '''
        raise NotImplementedError("Not Implemented")

    def get_adjacency_matrix(self, epsilon):
        '''
        This method returns the adjacency matrix for given epsilon (kernel width)

        Inputs:
            epsilon (int): kernel width

        Output:
            2d numpy array which can directly be used with plt.imshow(...) .
        '''
        raise NotImplementedError("Not Implemented")

    def get_best_epsilon(self):
        '''
        Returns the best epsilon for ISOMAP.
        This could be a hardcoded value or a strategy implemented by code.
        '''

        raise NotImplementedError("Not Implemented")


    def isomap(self, epsilon):
        '''
        Returns the first 2 principal components for the low embedding space.

        Inputs:
            epsilon (int): kernel width

        Output:
            (m, 2) numpy array.
        '''
        raise NotImplementedError("Not Implemented")

    def pca(self):
        '''
        Returns the first 2 principal components for the low embedding space.

        Output:
            (m, 2) numpy array .
        '''
        raise NotImplementedError("Not Implemented")
