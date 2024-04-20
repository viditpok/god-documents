import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):
    def __init__(self, X, K, max_iters=100):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K

    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        e_x = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        max_logit = np.max(logit, axis=1, keepdims=True)
        exp_logit = np.exp(logit - max_logit)
        sum_exp = np.sum(exp_logit, axis=1, keepdims=True)
        return np.log(sum_exp) + max_logit

    def normalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        variances = np.diagonal(sigma_i)
        differences = points - mu_i
        squared_differences = np.square(differences) / variances
        normalization_constant = 1 / np.sqrt(2 * np.pi * variances)
        pdf = normalization_constant * np.exp(-0.5 * squared_differences)
        return np.prod(pdf, axis=1)

    def multinormalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. Note the value in self.D may be outdated and not correspond to the current dataset.
            3. You may wanna check if the matrix is singular before implementing calculation process.
        """
        try:
            inv_sigma_i = np.linalg.inv(sigma_i)
        except np.linalg.LinAlgError:
            inv_sigma_i = np.linalg.inv(
                sigma_i + np.eye(sigma_i.shape[0]) * SIGMA_CONST
            )
        det_sigma_i = np.linalg.det(sigma_i)

        differences = points - mu_i
        exponent = -0.5 * np.sum(differences @ inv_sigma_i * differences, axis=1)
        normalization_constant = 1 / np.sqrt((2 * np.pi) ** mu_i.shape[0] * det_sigma_i)
        normal_pdf = normalization_constant * np.exp(exponent)
        return normal_pdf

    def create_pi(self):
        """
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        pi = np.full(self.K, 1 / self.K)
        return pi

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        indices = np.random.choice(self.N, self.K, replace=False)
        mu = self.points[indices]
        return mu

    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        sigma = np.array([np.eye(self.D) for _ in range(self.K)])
        return sigma

    def _init_components(self, **kwargs):
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5)
        pi = self.create_pi()
        mu = self.create_mu()
        sigma = self.create_sigma()
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        ll = np.zeros((self.N, self.K))
        for k in range(self.K):
            if full_matrix:
                pdf_values = self.multinormalPDF(self.points, mu[k], sigma[k])
            else:
                pdf_values = self.normalPDF(self.points, mu[k], sigma[k])
            ll[:, k] = np.log(pi[k] + LOG_CONST) + np.log(pdf_values + LOG_CONST)
        return ll

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        ll_joint = self._ll_joint(pi, mu, sigma, full_matrix)
        logsumexp_values = self.logsumexp(ll_joint)
        log_gamma = ll_joint - logsumexp_values
        gamma = self.softmax(log_gamma)
        return gamma

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        Nk = np.sum(gamma, axis=0) + LOG_CONST
        pi = Nk / self.N
        mu = (gamma.T @ self.points) / Nk[:, None]
        
        sigma = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            X_centered = self.points - mu[k]
            if full_matrix:
                sigma[k] = (gamma[:, k][:, np.newaxis] * X_centered).T @ X_centered / Nk[k]
            else:
                variances = (gamma[:, k] * (X_centered ** 2).T).sum(axis=1) / Nk[k]
                sigma[k] = np.diag(variances)
        
        return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the parameters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:

            gamma = self._E_step(pi, mu, sigma, full_matrix)

            pi, mu, sigma = self._M_step(gamma, full_matrix)

            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))
        return gamma, (pi, mu, sigma)
