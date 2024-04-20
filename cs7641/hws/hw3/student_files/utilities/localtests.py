import unittest
import numpy as np
from imgcompression import ImgCompression
from logistic_regression import LogisticRegression
from pca import PCA
from regression import Regression
from svd_recommender import SVDRecommender
from utilities.local_tests_folder.ic_test import IC_Test
from utilities.local_tests_folder.lr_test import LogisticRegression_Test
from utilities.local_tests_folder.pca_test import PCA_Test
from utilities.local_tests_folder.regression_test import Regression_Test
from utilities.local_tests_folder.svd_recommender_test import SVDRecommender_Test


def print_success_message(msg):
    print(f'UnitTest passed successfully for "{msg}"!')


class TestImgCompression(unittest.TestCase):
    """
    Tests for Q1: Image Compression
    """

    def test_svd_bw(self):
        """
        Test correct implementation of SVD calculation for black and white images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        U, S, V = ic.svd(test_ic.bw_image)
        self.assertEqual(np.allclose(U, test_ic.Ug), True, 'U is incorrect')
        self.assertEqual(np.allclose(S, test_ic.Sg), True, 'S is incorrect')
        self.assertEqual(np.allclose(V, test_ic.Vg), True, 'V is incorrect')
        success_msg = 'SVD calculation - black and white images'
        print_success_message(success_msg)

    def test_compress_bw(self):
        """
        Test correct implementation of image compression for black and white images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        U, S, V = test_ic.Ug, test_ic.Sg, test_ic.Vg
        Uc, Sc, Vc = ic.compress(U, S, V, 2)
        self.assertEqual(np.allclose(Uc, test_ic.Ugc), True,
            'U compression is incorrect')
        self.assertEqual(np.allclose(Sc, test_ic.Sgc), True,
            'S compression is incorrect')
        self.assertEqual(np.allclose(Vc, test_ic.Vgc), True,
            'V compression is incorrect')
        success_msg = 'Image compression - black and white images'
        print_success_message(success_msg)

    def test_rebuild_svd_bw(self):
        """
        Test correct implementation of SVD reconstruction for black and white images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        Xrebuild_g = ic.rebuild_svd(test_ic.Ugc, test_ic.Sgc, test_ic.Vgc)
        self.assertEqual(np.allclose(Xrebuild_g, test_ic.Xrebuild_g), True,
            'Reconstruction is incorrect')
        success_msg = 'SVD reconstruction - black and white images'
        print_success_message(success_msg)

    def test_compression_ratio_bw(self):
        """
        Test correct implementation of compression ratio calculation for black and white images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        cr = ic.compression_ratio(test_ic.bw_image, 2)
        self.assertEqual(np.allclose(cr, test_ic.cr_g), True,
            'Compression ratio is incorrect')
        success_msg = 'Compression ratio - black and white images'
        print_success_message(success_msg)

    def test_recovered_variance_proportion_bw(self):
        """
        Test correct implementation of recovered variance proportion calculation for black and white images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        rvp = ic.recovered_variance_proportion(test_ic.Sg, 2)
        self.assertEqual(np.allclose(rvp, test_ic.rvp_g), True,
            'Recovered variance proportion is incorrect')
        success_msg = 'Recovered variance proportion - black and white images'
        print_success_message(success_msg)

    def test_svd_color(self):
        """
        Test correct implementation of SVD calculation for color images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        U, S, V = ic.svd(test_ic.color_image)
        self.assertEqual(np.allclose(U, test_ic.Uc), True, 'U is incorrect')
        self.assertEqual(np.allclose(S, test_ic.Sc), True, 'S is incorrect')
        self.assertEqual(np.allclose(V, test_ic.Vc), True, 'V is incorrect')
        success_msg = 'SVD calculation - color images'
        print_success_message(success_msg)

    def test_compress_color(self):
        """
        Test correct implementation of image compression for color images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        U, S, V = test_ic.Uc, test_ic.Sc, test_ic.Vc
        Uc, Sc, Vc = ic.compress(U, S, V, 2)
        self.assertEqual(np.allclose(Uc, test_ic.Ucc), True,
            'U compression is incorrect')
        self.assertEqual(np.allclose(Sc, test_ic.Scc), True,
            'S compression is incorrect')
        self.assertEqual(np.allclose(Vc, test_ic.Vcc), True,
            'V compression is incorrect')
        success_msg = 'Image compression - color images'
        print_success_message(success_msg)

    def test_rebuild_svd_color(self):
        """
        Test correct implementation of SVD reconstruction for color images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        Xrebuild_c = ic.rebuild_svd(test_ic.Ucc, test_ic.Scc, test_ic.Vcc)
        self.assertEqual(np.allclose(Xrebuild_c, test_ic.Xrebuild_c), True,
            'Reconstruction is incorrect')
        success_msg = 'SVD reconstruction - color images'
        print_success_message(success_msg)

    def test_compression_ratio_color(self):
        """
        Test correct implementation of compression ratio calculation for color images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        cr = ic.compression_ratio(test_ic.color_image, 2)
        self.assertEqual(np.allclose(cr, test_ic.cr_c), True,
            'Compression ratio is incorrect')
        success_msg = 'Compression ratio - color images'
        print_success_message(success_msg)

    def test_recovered_variance_proportion_color(self):
        """
        Test correct implementation of recovered variance proportion calculation for color images
        """
        ic = ImgCompression()
        test_ic = IC_Test()
        rvp = ic.recovered_variance_proportion(test_ic.Sc, 2)
        self.assertEqual(np.allclose(rvp, test_ic.rvp_c), True,
            'Recovered variance proportion is incorrect')
        success_msg = 'Recovered variance proportion - color images'
        print_success_message(success_msg)


class TestSVDRecommender(unittest.TestCase):
    """
    Tests for Q1: SVD Recommender
    """

    def test_recommender_svd(self):
        """
        Test
        """
        recommender = SVDRecommender()
        test_recommender = SVDRecommender_Test()
        R, _, _ = recommender.create_ratings_matrix(test_recommender.ratings_df
            )
        U_k, V_k = recommender.recommender_svd(R, 10)
        my_slice_U_k, my_slice_V_k = test_recommender.get_slice_UV(U_k, V_k)
        correct_slice_U_k, correct_slice_V_k = (test_recommender.slice_U_k,
            test_recommender.slice_V_k)
        self.assertTrue(np.all(U_k.shape == test_recommender.
            U_k_expected_shape),
            'recommender_svd() function returning incorrect U_k shape')
        self.assertTrue(np.all(V_k.shape == test_recommender.
            V_k_expected_shape),
            'recommender_svd() function returning incorrect V_k shape')
        self.assertEqual(np.allclose(my_slice_U_k, correct_slice_U_k), True,
            'recommender_svd() function returning incorrect U_k')
        self.assertEqual(np.allclose(my_slice_V_k, correct_slice_V_k), True,
            'recommender_svd() function returning incorrect V_k')
        success_msg = 'recommender_svd() function'
        print_success_message(success_msg)

    def test_predict(self):
        """
        Test
        """
        recommender = SVDRecommender()
        recommender.load_movie_data()
        test_recommender = SVDRecommender_Test()
        R, users_index, movies_index = recommender.create_ratings_matrix(
            test_recommender.complete_ratings_df)
        mask = np.isnan(R)
        masked_array = np.ma.masked_array(R, mask)
        r_means = np.array(np.mean(masked_array, axis=0))
        R_filled = masked_array.filled(r_means)
        R_filled = R_filled - r_means
        U_k, V_k = recommender.recommender_svd(R_filled, k=8)
        movie_recommendations = recommender.predict(R, U_k, V_k,
            users_index, movies_index, test_recommender.test_user_id,
            test_recommender.movies_pool)
        print('Top 3 Movies the User would want to watch:')
        for movie in movie_recommendations:
            print(movie)
        print('--------------------------------------------------------------')
        self.assertEqual(len(movie_recommendations) == len(test_recommender
            .predict_expected_outputs), True,
            'predict() function is not returning the correct number of recommendations'
            )
        self.assertEqual((movie_recommendations == test_recommender.
            predict_expected_outputs).all(), True,
            'predict() function is not returning the correct recommendations')
        success_msg = 'predict() function'
        print_success_message(success_msg)


class TestPCA(unittest.TestCase):
    """
    Tests for Q2: PCA
    """

    def test_pca(self):
        """
        Test correct implementation of PCA
        """
        pca = PCA()
        test_pca = PCA_Test()
        pca.fit(test_pca.data)
        U, S, V = pca.U, pca.S, pca.V
        self.assertEqual(np.allclose(U, test_pca.U), True, 'U is incorrect')
        self.assertEqual(np.allclose(S, test_pca.S), True, 'S is incorrect')
        self.assertEqual(np.allclose(V, test_pca.V), True, 'V is incorrect')
        success_msg = 'PCA fit'
        print_success_message(success_msg)

    def test_transform(self):
        """
        Test correct implementation of PCA transform
        """
        pca = PCA()
        test_pca = PCA_Test()
        pca.fit(test_pca.data)
        U, S, V = pca.U, pca.S, pca.V
        X_new = pca.transform(test_pca.data)
        self.assertEqual(np.allclose(X_new, test_pca.X_new), True,
            'Transformed data is incorrect')
        success_msg = 'PCA transform'
        print_success_message(success_msg)

    def test_transform_rv(self):
        """
        Test correct implementation of PCA transform with recovered variance
        """
        pca = PCA()
        test_pca = PCA_Test()
        pca.fit(test_pca.data)
        U, S, V = pca.U, pca.S, pca.V
        X_new_rv = pca.transform_rv(test_pca.data, 0.7)
        self.assertEqual(np.allclose(X_new_rv, test_pca.X_new_rv), True,
            'Transformed data is incorrect')
        success_msg = 'PCA transform with recovered variance'
        print_success_message(success_msg)


class TestRegression(unittest.TestCase):
    """
    Tests for Q3: Regression
    """

    def test_rmse(self):
        """
        Test correct implementation of linear regression rmse
        """
        reg = Regression()
        test_reg = Regression_Test()
        rmse_test = np.allclose(reg.rmse(test_reg.predict, test_reg.y_all),
            test_reg.rmse)
        self.assertTrue(rmse_test, 'RMSE is incorrect')
        success_msg = 'RMSE'
        print_success_message(success_msg)

    def test_construct_polynomial_feats(self):
        """
        Test correct implementation of polynomial feature construction
        """
        reg = Regression()
        test_reg = Regression_Test()
        poly_feat_test = np.allclose(reg.construct_polynomial_feats(
            test_reg.x_all, 2), test_reg.construct_poly)
        self.assertTrue(poly_feat_test, 'Polynomial features are incorrect')
        success_msg = 'Polynomial feature construction'
        print_success_message(success_msg)

    def test_predict(self):
        """
        Test correct implementation of linear regression prediction
        """
        reg = Regression()
        test_reg = Regression_Test()
        predict_test = np.allclose(reg.predict(test_reg.x_all_feat,
            test_reg.true_weight), test_reg.predict)
        self.assertTrue(predict_test, 'Prediction is incorrect')
        success_msg = 'Linear regression prediction'
        print_success_message(success_msg)

    def test_linear_fit_closed(self):
        """
        Test correct implementation of closed form linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_closed_test = np.allclose(reg.linear_fit_closed(test_reg.
            x_all_feat, test_reg.y_all), test_reg.linear_closed, rtol=0.0001)
        self.assertTrue(linear_closed_test, 'Weights are incorrect')
        success_msg = 'Closed form linear regression'
        print_success_message(success_msg)

    def test_linear_fit_GD(self):
        """
        Test correct implementation of gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_GD, linear_GD_loss = reg.linear_fit_GD(test_reg.x_all_feat,
            test_reg.y_all)
        lgd_test = np.allclose(linear_GD, test_reg.linear_GD)
        lgd_loss_test = np.allclose(linear_GD_loss, test_reg.linear_GD_loss)
        self.assertTrue(lgd_test, 'Weights are incorrect')
        self.assertTrue(lgd_loss_test, 'Loss is incorrect')
        success_msg = 'Gradient descent linear regression'
        print_success_message(success_msg)

    def test_linear_fit_SGD(self):
        """
        Test correct implementation of stochastic gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_SGD, linear_SGD_loss = reg.linear_fit_SGD(test_reg.
            x_all_feat, test_reg.y_all, 1)
        lsgd_test = np.allclose(linear_SGD, test_reg.linear_SGD)
        lsgd_loss_test = np.allclose(linear_SGD_loss, test_reg.linear_SGD_loss)
        self.assertTrue(lsgd_test, 'Weights are incorrect')
        self.assertTrue(lsgd_loss_test, 'Loss is incorrect')
        success_msg = 'Stochastic gradient descent linear regression'
        print_success_message(success_msg)

    def test_ridge_fit_closed(self):
        """
        Test correct implementation of closed form ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_closed_test = np.allclose(reg.ridge_fit_closed(test_reg.
            x_all_feat, test_reg.y_all, 10), test_reg.ridge_closed)
        self.assertTrue(ridge_closed_test, 'Weights are incorrect')
        success_msg = 'Closed form ridge regression'
        print_success_message(success_msg)

    def test_ridge_fit_GD(self):
        """
        Test correct implementation of gradient descent ridge regression
        """
        error_atolerance = 1e-10
        reg = Regression()
        test_reg = Regression_Test()
        ridge_GD, ridge_GD_loss = reg.ridge_fit_GD(test_reg.x_all_feat,
            test_reg.y_all, 50000, 10)
        rgd_test = np.allclose(ridge_GD, test_reg.ridge_GD, atol=
            error_atolerance)
        rgd_loss_test = np.allclose(ridge_GD_loss, test_reg.ridge_GD_loss,
            atol=error_atolerance)
        rsgd_bias_incorrect = np.allclose(ridge_GD, test_reg.
            ridge_GD_bias_incorrect, atol=error_atolerance)
        self.assertFalse(rsgd_bias_incorrect,
            'Weights are incorrect. Make sure that you handle the bias term correctly.'
            )
        self.assertTrue(rgd_test, 'Weights are incorrect')
        self.assertTrue(rgd_loss_test, 'Loss is incorrect')
        success_msg = 'Gradient descent ridge regression'
        print_success_message(success_msg)

    def test_ridge_fit_SGD(self):
        """
        Test correct implementation of stochastic gradient descent ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_SGD, ridge_SGD_loss = reg.ridge_fit_SGD(test_reg.x_all_feat,
            test_reg.y_all, 20, 1)
        rsgd_test = np.allclose(ridge_SGD, test_reg.ridge_SGD)
        rsgd_loss_test = np.allclose(ridge_SGD_loss, test_reg.ridge_SGD_loss)
        rsgd_bias_incorrect = np.allclose(ridge_SGD, test_reg.
            ridge_SGD_bias_incorrect)
        self.assertFalse(rsgd_bias_incorrect,
            'Weights are incorrect. Make sure that you handle the bias term correctly.'
            )
        self.assertTrue(rsgd_test, 'Weights are incorrect')
        self.assertTrue(rsgd_loss_test, 'Loss is incorrect')
        success_msg = 'Stochastic gradient descent ridge regression'
        print_success_message(success_msg)

    def test_ridge_cross_validation(self):
        """
        Test correct implementation of ridge regression cross validation
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_cv_test = np.allclose(reg.ridge_cross_validation(test_reg.
            x_all_feat, test_reg.y_all, 3), test_reg.cross_val)
        self.assertTrue(ridge_cv_test, 'Weights are incorrect')
        success_msg = 'Ridge regression cross validation'
        print_success_message(success_msg)


class TestLogisticRegression(unittest.TestCase):
    """
    Tests for Q4: Logistic Regression
    """

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            sigmoid_result_slice), 'sigmoid incorrect')
        success_msg = 'Logistic Regression sigmoid'
        print_success_message(success_msg)

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        self.assertTrue(result.shape == test_lr.s.shape,
            'sigmoid incorrect: check shape')
        result_slice = result[:4, :4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            sigmoid_result_slice), 'sigmoid incorrect')
        success_msg = 'Logistic Regression sigmoid'
        print_success_message(success_msg)

    def test_bias_augment(self):
        """
        Test correct implementation of bias_augment
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.bias_augment(test_lr.x)
        result_slice_sum = np.sum(result[:4, :4])
        self.assertTrue(np.allclose(result_slice_sum, test_lr.
            bias_augment_slice_sum), 'bias_augment incorrect')
        success_msg = 'Logistic Regression bias_augment'
        print_success_message(success_msg)

    def test_predict_probs(self):
        """
        Test correct implementation of predict_probs
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_probs(test_lr.x_aug, test_lr.theta)
        self.assertTrue(result.ndim == 2,
            'predict_probs incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.x_aug.shape[0],
            'predict_probs incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            predict_probs_result_slice), 'predict_probs incorrect')
        success_msg = 'Logistic Regression predict_probs'
        print_success_message(success_msg)

    def test_predict_labels(self):
        """
        Test correct implementation of predict_labels
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_labels(test_lr.h_x)
        self.assertTrue(result.ndim == 2,
            'predict_labels incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.h_x.shape[0],
            'predict_labels incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            predict_labels_result_slice), 'predict_labels incorrect')
        success_msg = 'Logistic Regression predict_labels'
        print_success_message(success_msg)

    def test_loss(self):
        """
        Test correct implementation of loss
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.loss(test_lr.y, test_lr.h_x)
        self.assertAlmostEqual(result, test_lr.loss_result, msg=
            'loss incorrect')
        success_msg = 'Logistic Regression loss'
        print_success_message(success_msg)

    def test_gradient(self):
        """
        Test correct implementation of gradient
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.gradient(test_lr.x_aug, test_lr.y, test_lr.h_x)
        self.assertTrue(result.ndim == 2, 'gradient incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.x_aug.shape[1],
            'gradient incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.
            gradient_result_slice), 'gradient incorrect')
        success_msg = 'Logistic Regression gradient'
        print_success_message(success_msg)

    def test_accuracy(self):
        """
        Test correct implementation of accuracy
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.accuracy(test_lr.y, test_lr.y_hat)
        self.assertAlmostEqual(result, test_lr.accuracy_result,
            'accuracy incorrect')
        success_msg = 'Logistic Regression accuracy'
        print_success_message(success_msg)

    def test_evaluate(self):
        """
        Test correct implementation of evaluate
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.evaluate(test_lr.x, test_lr.y, test_lr.theta)
        self.assertAlmostEqual(result[0], test_lr.evaluate_result[0], msg=
            'evaluate incorrect')
        self.assertAlmostEqual(result[1], test_lr.evaluate_result[1], msg=
            'evaluate incorrect')
        success_msg = 'Logistic Regression evaluate'
        print_success_message(success_msg)

    def test_fit(self):
        """
        Test correct implementation of fit
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.fit(test_lr.x, test_lr.y, test_lr.x, test_lr.y, test_lr
            .lr, test_lr.epochs)
        self.assertTrue(result.ndim == 2, 'fit incorrect: check shape')
        self.assertTrue(result.shape[0] == test_lr.theta.shape[0],
            'fit incorrect: check shape')
        result_slice = result[:4]
        self.assertTrue(np.allclose(result_slice, test_lr.fit_result_slice),
            'fit incorrect')
        success_msg = 'Logistic Regression fit'
        print_success_message(success_msg)
