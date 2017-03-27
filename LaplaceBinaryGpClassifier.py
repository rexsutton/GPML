#!/usr/bin/env python
"""
   !!! Not certified fit for any purpose, use at your own risk !!!

   Copyright (c) Rex Sutton 2016.

   Python implementation of binary classification (machine learning),
       using a Gaussian Process with Laplace's approximation for the posterior.

   See `Rasmussen, Williams 2006 Gaussian Processes For Machine Learning'
       http://www.gaussianprocess.org/gpml/.

   The USPS data is at
       http://www.gaussianprocess.org/gpml/data/.

"""
import pickle
import numpy as np

from scipy.optimize import minimize
from scipy import linalg

from Tools import log_string
from Tools import print_vector_shape
from Tools import print_matrix_shape

__precision__ = np.float64
__tol__ = 0.00000001
__max_iters__ = 100000


def dot_matrix_diag(matrix, diag):
    """ Dot product of matrix wih diagonal matrix.
    Args:
        matrix (matrix): The matrix.
        diag (vector): Vector containing the diagonal.
    Returns:
        The dot product
    """
    return diag*matrix


def dot_diag_matrix(diag, matrix):
    """ Dot product of diagonal matrix with matrix.
    Args:
        diag (vector): Vector containing the diagonal.
        matrix (matrix): The matrix.
    Returns:
        The dot product
    """
    return (diag*matrix.T).T


def add_diag(matrix, constant):
    """ Add a constant to each entry in the matrix diagonal.
    Args:
        matrix (matrix): The matrix.
        constant (real): The real number.
    Returns:
        The modified matrix.
    """
    np.fill_diagonal(matrix, matrix.diagonal() + constant)


def dot_factored_matrix(l_matrix, rhs):
    """ Equivalent to multiplying rhs vector (or matrix)
        by the inverse of the matrix with Cholesky factorization L.
    Args:
        l_matrix (matrix): The Cholesky factored matrix.
        rhs (vector matrix): The vector or matrix.
    Returns:
        Vector or matrix
    """
    temp = linalg.solve_triangular(l_matrix, rhs, lower=True)
    return linalg.solve_triangular(l_matrix, temp, lower=True, trans=1)


class SquaredExponentialKernel(object):
    """The squared exponential kernel.
        A functor object for computing the value of the squared exponential
            kernel for observations.
    """

    def __init__(self):
        """Initialise the kernel functor.
        """
        self.log_sigma_f = None
        self.log_l = None
        self.sigma_f_squared = None
        self.two_sigma_f_squared = None
        self.neg_two_l_squared = None
        self.two_log_sigma_f = None

    @staticmethod
    def num_params():
        """Return the number of parameters.
        """
        return 2

    def set_params(self, params):
        """Initialise the kernel functor with parameter values.
        Args:
            params: The parameters.
        """
        self.log_sigma_f = params[0]
        self.log_l = params[1]
        self.sigma_f_squared = np.exp(2.0 * self.log_sigma_f)
        self.two_sigma_f_squared = 2.0 * self.sigma_f_squared
        self.neg_two_l_squared = -2.0 * np.exp(2.0 * self.log_l)
        self.two_log_sigma_f = 2.0 * self.log_sigma_f

    def compute(self, lhs, rhs, compute_diag, compute_derivatives):
        """Compute the kernel and derivatives for observation vectors, assumed to be same length.
        Args:
            lhs (vector): The first observation vector.
            rhs (vector): The second observation vector.
            compute_diag(bool): Hint to indicate the rhs is the lhs.
            compute_derivatives(bool): If True compute derivatives.
        Returns:
             The kernel value and derivatives with respect to the hyper-parameters.
        """
        derivatives = np.empty([2], __precision__)

        if compute_diag:
            kernel = self.sigma_f_squared

            if compute_derivatives:
                derivatives[0] = self.two_sigma_f_squared
                derivatives[1] = 0.0
        else:
            tmp = lhs - rhs
            magnitude_term = np.dot(tmp, tmp) / self.neg_two_l_squared
            exponent = self.two_log_sigma_f + magnitude_term
            kernel = np.exp(exponent)

            if compute_derivatives:
                two_kernel = 2.0 * kernel
                derivatives[0] = two_kernel
                derivatives[1] = -two_kernel * magnitude_term

        return kernel, derivatives


class KernelDecorator(object):
    """The kernel.
        A functor object for computing the value of the kernel for observations.
    """

    def __init__(self, kernel):
        """Initialise the kernel decorator with a kernel.
        Args:
            kernel (function): The kernel function.
        """
        self.__kernel__ = kernel

    def num_params(self):
        """Return the number of parameters.
        """
        return self.__kernel__.num_params()

    def set_params(self, params):
        """Initialise the kernel functor with parameter values.
        Args:
            params: The parameters.
        """
        self.__kernel__.set_params(params)

    def compute(self, lhs, rhs):
        """Compute the kernel for observation vectors, assumed to be same length.
        Args:
            lhs (vector): The first observation vector.
            rhs (vector): The second observation vector.
        """
        value, dummy =\
            self.__kernel__.compute(lhs, rhs, compute_diag=False, compute_derivatives=False)
        return value

    def compute_covariance_vector(self, observation_matrix, observation):
        """Apply the kernel to each pair of observations formed from the matrix and the observation.
        Args:
            observation_matrix (matrix): The vector of N observations,
                first index is the number of observations.
            observation (vector): An observation.
        Returns:
            vector: The vector of kernel values.
        """
        num = len(observation_matrix[0])
        covariance_vector = np.empty([num], dtype=__precision__)
        for j in range(0, num):
            value = self.compute(observation_matrix[:, j], observation)
            covariance_vector[j] = value
        return covariance_vector

    def compute_covariance_matrix(self, observation_matrix):
        """Apply the kernel to each observation pair, and return derivatives.
        Args:
            observation_matrix (matrix): Matrix of N observations.
        Returns:
            matrix: The N by N matrix of kernel values.
        """
        num = len(observation_matrix[0])
        num_parms = self.__kernel__.num_params()
        # create covariance matrix
        covariance_matrix = np.empty([num, num], dtype=__precision__)
        # create derivatives matrices
        derivatives = []
        for idx_param in range(0, num_parms):
            derivatives.append(np.empty([num, num], dtype=__precision__))
        # for each pair of observations
        num = len(observation_matrix[0])
        for i in range(0, num):
            for j in range(i, num):
                if i == j:
                    # on the diagonal
                    observation = observation_matrix[:, i]
                    covariance, gradients \
                        = self.__kernel__.compute(observation,
                                                  observation,
                                                  compute_diag=True,
                                                  compute_derivatives=True)
                    covariance_matrix[i, i] = covariance
                    # set the derivatives
                    for idx_param in range(0, num_parms):
                        derivatives[idx_param][i, i] = gradients[idx_param]

                else:
                    # off diagonal
                    covariance, gradients \
                        = self.__kernel__.compute(observation_matrix[:, i],
                                                  observation_matrix[:, j],
                                                  compute_diag=False,
                                                  compute_derivatives=True)
                    # set covariance matrix
                    covariance_matrix[i, j] = covariance
                    covariance_matrix[j, i] = covariance
                    # set the derivatives
                    for idx_param in range(0, num_parms):
                        deriv = gradients[idx_param]
                        derivatives[idx_param][i, j] = deriv
                        derivatives[idx_param][j, i] = deriv
        # return
        return covariance_matrix, derivatives


class DerivedData(object):
    """A structure encapsulating the derived data.
    """
    def __init__(self, observation_vector, classification_vector, kernel):
        """Initialise the members of the derived data cache.
        Args:
            observation_vector (matrix): The vector of observations.
            classification_vector (vector): The vector of classifications .
            kernel (functor): The kernel function to use.
        """
        self.observation_vector = observation_vector
        self.classification_vector = classification_vector
        self.kernel = KernelDecorator(kernel)
        self.cov_matrix, self.deriv_matrices \
            = self.kernel.compute_covariance_matrix(self.observation_vector)

    def print_diagnosis(self):
        """Print information about the instance.
        """
        print_matrix_shape("X", self.observation_vector)
        print_vector_shape("y", self.classification_vector)
        print_matrix_shape("K", self.cov_matrix)
        # print derivatives
        for matrix in self.deriv_matrices:
            print_matrix_shape("dK", matrix)

    def save(self, path):
        """Save instance to `path'.
        Args:
            path (str): The path to save the derived data to.
        """
        with open(path, 'wb') as filestream:
            pickle.dump(self, filestream)

    @staticmethod
    def load(path):
        """Load instance from `path'.
        Args:
            path (str): The path to save the derived data to.
        """
        with open(path, 'rb') as filestream:
            return pickle.load(filestream)


def logistic(real):
    """The logistic function.
    Args:
        real (float): The parameter.
    Returns:
        float: The value of the logistic function.
    """
    return 1.0 / (1.0 + np.exp(-real))


def log_likelihood(y_vec, f_vec):
    """Compute the log probability of classification vector y given f.
    Args:
        y_vec (vector): The vector of classifications.
        f_vec (vector): The vector of the values of the latent function f,
            with classifications classification_vector.
    Returns:
        float: The log-likelihood of classification y_vec,
            given latent function values f_vec.
    """
    return np.sum(np.log(logistic(np.multiply(f_vec, y_vec))))


def log_likelihood_deriv(y_vec, f_vec):
    """Compute the first derivative of the log-likelihood.
    Args:
        y_vec (vector): The vector of classifications.
        f_vec (vector): The value of latent function f at x_i with classification y_i.
    Returns:
        vector: The derivative of log-likelihood of classification classification_vector
            w.r.t to latent function values f_i.
    """
    pi_vec = logistic(f_vec)
    t_vec = 0.5 * (y_vec + 1.0)
    return t_vec - pi_vec


def log_likelihood_second_deriv(f_vec):
    """Compute the second derivatives of the log-likelihood, only diagonal elements are non-zero.
    Args:
        f_vec (vector): The value of latent function f at x_i with classification y_i.
    Returns:
        vector: The diagonal of the matrix of second derivatives.
    """
    pi_vec = logistic(f_vec)
    opi_vec = pi_vec - 1.0
    diag_vec = np.multiply(pi_vec, opi_vec)
    return diag_vec


def log_likelihood_third_deriv(f_vec):
    """Compute the third derivatives of the log-likelihood, only diagonal elements are non-zero.
    Args:
        f_vec (vector): The value of latent function f at x_i with classification y_i.
    Returns:
        vector: The diagonal of the matrix of third derivatives.
    """
    pi_vec = logistic(f_vec)
    opi_vec = 1.0 - pi_vec
    pi_opi_vec = np.multiply(pi_vec, opi_vec)
    return np.multiply(pi_vec, pi_opi_vec) - np.multiply(opi_vec, pi_opi_vec)


def compute_r_matrix(w_sqrt_diag, l_matrix):
    """ Compute the matrix describing the posterior variance.
    Args:
        w_sqrt_diag (vector): The square root of the diagonal of the W matrix.
        l_matrix (matrix): The Cholesky factored matrix.
    Returns:
        The R matrix.
    """
    return dot_diag_matrix(w_sqrt_diag, dot_factored_matrix(l_matrix, np.diag(w_sqrt_diag)))


def find_fhat(initial_guess, cov_matrix, y_vec):
    """Find the mean of the approximate posterior using Newton's method.
    Args:
        initial_guess (vector): The initial guess for fhat.
        cov_matrix (matrix): The covariance matrix .
        y_vec (vector): The vector of classifications.
    Returns:
        A tuple, elements are:
         fhat,
         the `a vector' the first derivative of the log likelihood at fhat, K^-1 dot fhat.
         the `vec vector
    """
    f_vec = initial_guess
    a_vec = None
    w_sqrt_diag = None
    l_matrix = None
    # iterate until convergence
    num_iters = 0
    while True:
        # w matrix
        w_diag = -1.0 * log_likelihood_second_deriv(f_vec)
        w_sqrt_diag = np.sqrt(w_diag)
        # b vector
        b_vec = np.dot(np.diag(w_diag), f_vec) + log_likelihood_deriv(y_vec, f_vec)
        # B matrix
        b_matrix = dot_diag_matrix(w_sqrt_diag, dot_matrix_diag(cov_matrix, w_sqrt_diag))
        add_diag(b_matrix, 1.0)
        # L matrix
        l_matrix = np.linalg.cholesky(b_matrix)
        # a vector
        a_vec = \
            b_vec - dot_diag_matrix(w_sqrt_diag,
                                    dot_factored_matrix(l_matrix,
                                                        dot_diag_matrix(w_sqrt_diag,
                                                                        np.dot(cov_matrix, b_vec))))
        # f t+1
        f_new_vec = np.dot(cov_matrix, a_vec)
        num_iters += 1
        # test convergence
        delta_f_vec = f_new_vec - f_vec
        magnitude_delta = np.sqrt(np.dot(delta_f_vec, delta_f_vec))
        # print a warning if we have iterated for too long
        if num_iters == __max_iters__:
            print "!!! increase iters !!!, delta:", magnitude_delta
        # break out
        if num_iters == __max_iters__ or magnitude_delta < __tol__:
            break
        # didn't break so update
        f_vec = f_new_vec
    # return
    return f_vec, a_vec, w_sqrt_diag, l_matrix


def posterior_mean_for_fstar(a_vec, kstar):
    """Calculate the mean of the posterior distribution.
    Args:
        a_vec (vector): Intermediate quantity a_vec.
        kstar (vector): Intermediate quantity kstar.
    Returns:
        float: The mean of the posterior distribution.
    """
    return np.dot(np.transpose(kstar), a_vec)


def posterior_variance_for_fstar(r_matrix, kstar, kxstar):
    """Calculate the variance of the posterior distribution.
    Args:
        r_matrx (vector): Intermediate quantity r_matrix.
        kstar (vector): Intermediate quantity kstar.
        kxstar (vector): Intermediate quantity kxstar.
    Returns:
        float: The variance of the posterior distribution.
    """
    tmp = np.dot(np.transpose(kstar), np.dot(r_matrix, kstar))
    # floor variance
    floor = 0.00001
    variance = kxstar - tmp
    if variance < floor:
        variance = floor
    return variance


def hyper_param_deriv(covariance_matrix,
                      deriv_covariance_matrix,
                      a_vec,
                      r_matrix,
                      s2_vec):
    """Return the derivative of the log marginal likelihood.
    Args:
        covariance_matrix (matrix): The covariance matrix .
        deriv_covariance_matrix (matrix): The derivative of the covariance matrix .
        a_vec (vector): Intermediate quantity.
        r_matrix (matrix): Intermediate quantity.
        s2_vec(vector): Intermediate quantity.
    Returns:
        The derivative.
    """
    # the explicit term equation 5.22 p125(143)
    s1_real = \
        0.5 * np.dot(a_vec.transpose(), np.dot(deriv_covariance_matrix, a_vec)) \
        - 0.5 * np.trace(np.dot(r_matrix, deriv_covariance_matrix))
    # NOTE we use a_vec is the first derivative of the log-likelihood at f_hat here
    b_vec = np.dot(deriv_covariance_matrix, a_vec)
    # equation 5.24 p125(143)
    s3_vec = b_vec - np.dot(covariance_matrix, np.dot(r_matrix, b_vec))
    # print_vector_shape("s3_vec", s3_vec)
    return s1_real + np.dot(s2_vec, s3_vec)


class TrainingObjective(object):  # pylint: disable=too-few-public-methods
    """Objective function for training, provides `warm-start' for fhat.
    """
    def __init__(self, patterns, classifications, kernel):
        """Memberwise initialisation.
        Args:
            patterns (matrix): The training observations.
            classifications (vector): The vector of classifications.
            kernel (function): The kernel function.
        Returns:
            The gradients of the quantity being minimised.
        """
        self.fhat = None
        self.patterns = patterns
        self.classifications = classifications
        self.kernel = kernel

    def __call__(self, params):
        """Objective function and gradients for training the model.
        Args:
            params (vector): The kernel parameters .
        Returns:
            The objective function and gradients for training the model.
        """
        self.kernel.set_params(params)
        data = DerivedData(self.patterns, self.classifications, self.kernel)
        pred = Classifier(data, self.fhat)
        # cache fhat as the initial guess for subsequent invocations.
        self.fhat = pred.fhat
        value = pred.log_marginal_likelihood()
        derivatives = pred.log_marginal_likelihood_deriv()
        print log_string(), "params:", params, "objective:", value, "derivatives:", derivatives
        return -1.0 * value, -1.0 * derivatives


class Classifier(object):
    """A structure encapsulating the derived data and state required
        to make predictions.
    """

    def __init__(self, data, fhat_initial_guess=None):
        """Initialise by performing long calculations.
        Args:
            data (DerivedData): The derived data.
            fhat_initial_guess (vector): The initial guess for fhat.
        """
        self.data = data
        self.piovereight = 0.39269908169872414
        # set initial guess
        if fhat_initial_guess is None:
            fhat_initial_guess = np.zeros(len(self.data.classification_vector))
        # find the mean of the posterior
        self.fhat, self.a_vec, self.w_sqrt_diag, self.l_matrix \
            = find_fhat(fhat_initial_guess, self.data.cov_matrix, self.data.classification_vector)
        # compute derived quantity
        self.r_matrix = compute_r_matrix(self.w_sqrt_diag, self.l_matrix)

    def predict(self, pattern):
        """The probability of positive classification for the pattern.
        Args:
            pattern (vector): An observation.
        Returns:
            float: The probability of positive classification for xstar.
        """
        kstar = self.data.kernel.compute_covariance_vector(self.data.observation_vector, pattern)
        kxstar = self.data.kernel.compute(pattern, pattern)
        mean = posterior_mean_for_fstar(self.a_vec, kstar)
        variance = posterior_variance_for_fstar(self.r_matrix, kstar, kxstar)
        # Mackay's approximation, fastest
        kappa = np.sqrt(1.0 / (1.0 + self.piovereight * variance))
        return logistic(mean * kappa)

    @staticmethod
    def threshold(probability):
        """Return the classification for probability.
        Args:
            probability (float): The probability.
        Returns:
            float: The classification.
        """
        return 1.0 if probability >= 0.5 else -1.0

    def classify(self, pattern):
        """Return the classification for pattern.
        Args:
            pattern (vector): An observation.
        Returns:
            float: The classification for pattern.
        """
        return self.threshold(self.predict(pattern))

    def map_classify(self, pattern):
        """Return the Maximum-A-Priori classification for pattern (gives same result as classify).
        Args:
            pattern (vector): An observation.
        Returns:
            float: The probability of positive classification for pattern.
        """
        kstar = self.data.kernel.compute_covariance_vector(self.data.observation_vector, pattern)
        mean = posterior_mean_for_fstar(self.a_vec, kstar)
        return self.threshold(logistic(mean))

    def log_marginal_likelihood(self):
        """Return the log marginal likelihood of the data.
        Returns:
            The log marginal likelihood.
        """
        ret = -0.5 * np.dot(self.a_vec, self.fhat) \
                + log_likelihood(self.data.classification_vector, self.fhat) \
                - np.sum(np.log(self.l_matrix.diagonal()))
        return ret

    def log_marginal_likelihood_deriv(self):
        """Return the derivatives of the log marginal likelihood.
        Returns:
            The log marginal likelihood and vector of derivatives.
        """
        # C matrix
        c_matrix = linalg.solve_triangular(self.l_matrix,
                                           dot_diag_matrix(self.w_sqrt_diag, self.data.cov_matrix),
                                           lower=True)
        # s2 vector
        # NOTE the textbook is wrong,
        #  there is no minus sign as the W = - nabla_nabla log likelihood.
        temp = self.data.cov_matrix.diagonal() - np.dot(c_matrix.transpose(), c_matrix).diagonal()
        s2_vec = 0.5 * np.multiply(temp, log_likelihood_third_deriv(self.fhat))
        # for each hyper-parameter
        ret = np.empty([len(self.data.deriv_matrices)])
        for index, deriv_matrix in enumerate(self.data.deriv_matrices):
            # calculate the derivative
            ret[index] = \
                hyper_param_deriv(self.data.cov_matrix,
                                  deriv_matrix,
                                  self.a_vec,
                                  self.r_matrix,
                                  s2_vec)
        # return
        return ret


    @staticmethod
    def train(kernel, patterns, classifications):
        """Train the classifier by finding parameter values.
        Args:
            kernel (function): The kernel function.
            patterns (matrix): The vector of training observations.
            classifications (vector): The vector of classifications.
        Returns:
            vector: The hyper-parameter values.
        """
        res = minimize(TrainingObjective(patterns, classifications, kernel),
                       np.zeros(kernel.num_params()),
                       jac=True)
        return res.x

    def print_diagnosis(self):
        """Print information about the instance.
        """
        print_vector_shape("fhat:", self.fhat)
        print_vector_shape("a_vec:", self.a_vec)
        print_vector_shape("w_sqrt_diag:", self.w_sqrt_diag)
        print_matrix_shape("l_matrix:", self.l_matrix)
        print_matrix_shape("r_matrix:", self.r_matrix)
        self.data.print_diagnosis()

    def save(self, path):
        """Save instance to `path'.
        Args:
            path (str): The path to save the derived data to.
        """
        with open(path, 'wb') as filestream:
            pickle.dump(self, filestream)

    @staticmethod
    def load(path):
        """Load instance from a local file `predictor'.
        Args:
            path (str): The path to save the derived data to.
        """
        with open(path, 'rb') as filestream:
            return pickle.load(filestream)
