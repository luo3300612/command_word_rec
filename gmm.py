import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.stats import multivariate_normal
from utils import multinomial_sample, logSumExp, elog
import scipy
from scipy.special import logsumexp, softmax


# def softmax(x):
#     out = np.exp(x)
#     out = out / np.sum(out, axis=1, keepdims=True)
#     # print(np.sum(out, axis=1).shape)
#     return out


class GMM:
    def __init__(self, K, n, weights=None, means=None, covars=None, init=True, covar_type='full', verbose=False):
        self.K = K  # num of gaussians
        self.n = n  # dim of gaussians
        self.means = means
        self.covars = covars
        self.weights = weights  # weight of each gaussian
        self.covar_type = covar_type
        self.reestimate_time = 0
        if init:
            self._self_init(verbose)

    def sample(self, N):
        # seems impossible
        pass

    def _self_init(self, verbose):
        """
        random init
        """
        if self.means is None:
            self.means = [np.random.randint(-10, 10, (self.n,)) for _ in range(self.K)]
        if self.covars is None:
            self.covars = [np.diag(np.random.randint(1, 2, (self.n,))) for _ in range(self.K)]
        if self.weights is None:
            self.weights = [1 / self.K for _ in range(self.K)]
        if verbose:
            print('init res:')
            print('mean:', self.mleans)
            print('covars:', self.covars)
            print('weights:', self.weights)

    def kmeans_init(self, data, tolerance=5, verbose=False):
        self.reestimate_time += 1
        nan_time = 0
        while True:
            belongings = kmeans(data, self.K, verbose=verbose)
            cov = np.cov(data.T)
            for i in range(self.K):
                cluster_data = data[belongings == i]
                self.means[i] = np.mean(cluster_data, axis=0)
                # self.covars[i] = np.cov(cluster_data.T)
                self.covars[i] = cov
                if self.covar_type == 'diag':
                    self.covars[i] = np.diag(np.diag(self.covars[i]))
                if verbose:
                    print('init gaussian{} to mean: {} cov: {}'.format(i, self.means[i], self.covars[i]))
                try:
                    assert not np.isnan(self.means[i]).any()
                    assert not np.isnan(self.covars[i]).any()
                except AssertionError:
                    print('k-means init error')
                    print('data info')
                    print(data.shape)
                    for j in range(self.K):
                        print('mean', j)
                        print(self.means[j])
                        print('covar', j)
                        print(self.covars[j])

            self.weights = [1 / self.K for _ in range(self.K)]
            if not self.nan_check():
                break
            else:
                nan_time += 1
                print('Nan exists, redo k-means, redo times: {}'.format(nan_time))
            if nan_time == tolerance:
                raise ValueError('Nan exists in init for {} times'.format(tolerance))

    def nan_check(self):
        for i in range(self.K):
            if np.isnan(self.means[i]).any():
                return True
            if np.isnan(self.covars[i]).any():
                return True
        return False

    # TODO strange position, should put out of the class
    def loglikelihood(self, point, mean, covar):
        # return 1 / ((2 * np.pi) ** self.n / 2) / (np.linalg.det(covar) ** 0.5) * np.e ** (
        #         -(point - mean).T.dot(np.linalg.inv(covar)).dot(point - mean) / 2)
        # return 0.5 * (-self.n * np.log(2 * np.pi) - np.log(np.linalg.det(covar)) - (point - mean).T.dot(
        #     np.linalg.inv(covar)).dot(point - mean))
        res = multivariate_normal.logpdf(point, mean, covar, allow_singular=True)
        res[np.isnan(res)] = -1e8
        return res

    def pdf(self, point):
        prob = 0
        for weight, mean, covar in zip(self.weights, self.means, self.covars):
            prob += weight * multivariate_normal.pdf(point, mean, covar)
        return prob

    def logpdf(self, point):
        prob = np.zeros((self.K, point.shape[0]))
        for i, (weight, mean, covar) in enumerate(zip(self.weights, self.means, self.covars)):
            prob[i] = np.log(weight) + multivariate_normal.logpdf(point, mean, covar, allow_singular=True)
            prob[i][np.isnan(prob[i])] = -1e8
            try:
                assert not np.isnan(prob[i]).any()
            except AssertionError:
                np.set_printoptions(formatter={'float': '{: 0.2e}'.format}, threshold=1000000)
                print('nan exists in logpdf')
                print('prob i')
                print(prob[i])
                print('weight')
                print(weight)
                print('mean')
                print(mean)
                print('covar')
                print(covar)
                raise
        prob = logsumexp(prob, axis=0)
        try:
            assert not np.isnan(prob.any())
        except AssertionError:
            np.set_printoptions(formatter={'float': '{: 0.2e}'.format}, threshold=1000000)
            print('nan exists after logsumexp')
            print('origin prob')
            print(prob)
            raise
        assert prob.shape == (point.shape[0],)
        return prob

    def reestimate(self, data):
        self.reestimate_time += 1
        N = data.shape[0]
        belongings = np.zeros((N, self.K))
        for j in range(self.K):
            mean, covar = self.means[j], self.covars[j]
            try:
                assert not np.isnan(mean).any()
                assert not np.isnan(covar).any()
            except AssertionError:
                print('Nan exists when re-estimate start')
                print('re estimate time')
                print(self.reestimate_time)
                print('data info')
                print(data.shape)
                for i in range(self.K):
                    print('mean', i)
                    print(self.means[i])
                    print('covar', i)
                    print(self.covars[i])
                raise
            belongings[:, j] = self.loglikelihood(data, mean, covar) + np.log(self.weights[j])

        normed_belongings = softmax(belongings, axis=-1)
        # m-step
        sum_normed_belongings = normed_belongings.sum(axis=0)
        new_weights = sum_normed_belongings / sum_normed_belongings.sum()
        self.weights = new_weights.tolist()

        for j in range(self.K):
            gamma = normed_belongings[:, j].reshape(-1, 1)
            if self.weights[j] < 1e-4:
                # do not update
                pass
                # or re-initialize
                # mean = np.mean(data, axis=0)
                # covar = np.cov(data.T)
                # print('warning: data info {}, weight for component < 1e-8, reinit'.format(data.shape))
            else:
                mean = np.sum(gamma * data, axis=0) / np.sum(gamma, axis=0, keepdims=True)
                covar = (gamma * (data - mean)).T.dot(data - mean) / np.sum(gamma, axis=0, keepdims=True)
                self.means[j] = mean.reshape(-1, )
                self.covars[j] = covar
            if self.covar_type == 'diag':
                self.covars[j] = np.diag(np.diag(self.covars[j]))
            try:
                assert not np.isnan(self.means[j]).any()
                assert not np.isnan(self.covars[j]).any()
            except AssertionError:
                print('Nan exists when re-estimate update')
                print('re estimate time')
                print(self.reestimate_time)
                print('data info')
                print(data.shape)
                print('belongings')
                print(belongings)
                print('normed belongings')
                print(normed_belongings)
                for i in range(self.K):
                    print('mean', i)
                    print(self.means[i])
                    print('covar', i)
                    print(self.covars[i])

    def train(self, data, step=100, eps=1e-4, kmeans_init=False, verbose=False):
        # data N * n
        if kmeans_init:
            self.kmeans_init(data, verbose=verbose)
        N = data.shape[0]
        n = data.shape[1]
        flag = False
        # init e-step
        belongings = np.zeros((N, self.K))
        for j in range(self.K):
            mean, covar = self.means[j], self.covars[j]
            try:
                assert not np.isnan(self.means[j]).any()
                assert not np.isnan(self.covars[j]).any()
            except AssertionError:
                print('after k-means before training')
                print('data info')
                print(data.shape)
                print('means')
                for i in range(self.K):
                    print('means', i)
                    print(self.means[i])
                print('covars')
                for i in range(self.K):
                    print('covars', i)
                    print(self.covars[j])
                raise
            belongings[:, j] = self.loglikelihood(data, mean, covar) + np.log(self.weights[j])
        last_loglikelihood = np.sum(logsumexp(belongings, axis=-1))
        if verbose:
            print('init, loglikelihood: {}'.format(last_loglikelihood))

        for i in range(step):
            normed_belongings = softmax(belongings, axis=-1)
            # m-step
            sum_normed_belongings = normed_belongings.sum(axis=0)
            new_weights = sum_normed_belongings / sum_normed_belongings.sum()
            self.weights = new_weights.tolist()

            for j in range(self.K):
                gamma = normed_belongings[:, j].reshape(-1, 1)
                mean = np.sum(gamma * data, axis=0) / np.sum(gamma, axis=0, keepdims=True)
                covar = (gamma * (data - mean)).T.dot(data - mean) / np.sum(gamma, axis=0, keepdims=True)
                self.means[j] = mean.reshape(-1, )
                self.covars[j] = covar
                if self.covar_type == 'diag':
                    self.covars[j] = np.diag(np.diag(self.covars[j]))

            # e-step
            belongings = np.zeros((N, self.K))
            for j in range(self.K):
                mean, covar = self.means[j], self.covars[j]
                belongings[:, j] = self.loglikelihood(data, mean, covar) + np.log(self.weights[j])
            current_loglikelihood = np.sum(logsumexp(belongings, axis=-1))
            if verbose:
                print('step {}, loglikelihood: {}'.format(i + 1, current_loglikelihood))

            delta = np.fabs(current_loglikelihood - last_loglikelihood)
            if delta < eps:
                if verbose:
                    print('eps reached, Done')
                flag = True
            last_loglikelihood = current_loglikelihood
        else:
            if verbose:
                print('step reached, eps={}, Done'.format(delta))
        if verbose:
            print('Done')
            print('Res means')
            print(self.means)
            print('Res covars')
            print(self.covars)
            print('Res weights')
            print(self.weights)
        return flag


def kmeans(X, K, eps=1e-4, verbose=False):
    # X m * n
    m = X.shape[0]
    n = X.shape[1]
    init_centers_indexes = random.sample(list(range(m)), K)
    centers = X[init_centers_indexes, :].reshape(K, 1, n)  # K * 1 * n

    last_loss = 0

    # e-step
    X = X.reshape(1, m, n)  # 1 * m * n
    distances = X - centers  # K * m * n
    distances = np.sum(distances * distances, axis=-1).T  # m * K
    belongings = np.argmin(distances, axis=1)  # m
    loss = np.sum(np.min(distances, axis=1)) / m

    while np.fabs(loss - last_loss) > eps:
        last_loss = loss
        # m-step
        for i in range(K):
            cluster_data = X.reshape(m, n)[belongings == i]
            centers[i] = np.mean(cluster_data, axis=0)

        # e-step
        distances = X - centers  # K * m * n
        distances = np.sum(distances * distances, axis=-1).T  # m * K
        belongings = np.argmin(distances, axis=1)  # m
        loss = np.sum(np.min(distances, axis=1)) / m
        if verbose:
            print('loss:{:.4f}'.format(loss))
    return belongings


if __name__ == '__main__':
    # 第一簇的数据
    num1, mu1, var1 = 400, [0.5, 0.5], [1, 3]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, [5.5, 2.5], [2, 2]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, [1, 7], [6, 2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # GMM test
    gmm = GMM(3, 2, covar_type='diag')
    gmm.train(X, 100, kmeans_init=True, verbose=True)

    # k-means test
    # belongings = kmeans(X, 3)
    # x1 = X[belongings == 0]
    # x2 = X[belongings == 1]
    # x3 = X[belongings == 2]
    #
    # plt.subplot(121)
    # plt.scatter(X1[:, 0], X1[:, 1], c='r')
    # plt.scatter(X2[:, 0], X2[:, 1], c='b')
    # plt.scatter(X3[:, 0], X3[:, 1], c='y')
    # plt.subplot(122)
    # plt.scatter(x1[:, 0], x1[:, 1], c='r')
    # plt.scatter(x2[:, 0], x2[:, 1], c='b')
    # plt.scatter(x3[:, 0], x3[:, 1], c='y')
    # plt.show()
