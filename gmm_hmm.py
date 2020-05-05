import numpy as np
from gmm import GMM
from typing import List
from utils import multinomial_sample, logSumExp, elog
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import h5py


class CHMM:
    """
    Continuous HMM
    """
    def __init__(self, N, A, GMM_kwargs, pi, state_map=None, observation_map=None):
        self.N = N  # kinds of states, int
        # self.M = M  # kinds of observed results, int
        self.A = np.array(A)  # state transferring probability, N * N matrix
        self.B = [GMM(**GMM_kwargs) for _ in range(self.N - 1)]  # pdf of observed results, for each state
        self.pi = pi  # pdf of init state, len N list
        self.K = self.B[0].K

        self.feats_lists = [[] for _ in range(N - 1)]
        self.state_transfer_historys = []
        if state_map is None:
            self.state_map = ['s' + str(i) for i in range(1, N + 1)]

    def sample(self, step=5):
        raise NotImplementedError
        # state_history = []
        # observation_history = []
        # if step is 0:
        #     return state_history, observation_history
        # state_history.append(multinomial_sample(self.pi))
        # observation_history.append(self.B[state_history[-1]].sample(1))
        # for t in range(step - 1):
        #     state_history.append(multinomial_sample(self.A[state_history[-1]]))
        #     observation_history.append(self.B[state_history[-1]].sample(1))
        # return state_history, observation_history

    def assign(self, features, states=None):
        # for viterbi training
        # if states is not given, average split features to each state
        assert features.shape[1] == self.B[0].n
        if states is None:
            for feats_list, feats in zip(self.feats_lists, np.array_split(features, self.N - 1)):
                feats_list.append(feats)
        else:
            np_state_split = [np.sum(np.array(states) == i) for i in range(self.N - 1)]
            ac_np_state_split = np.add.accumulate(np_state_split)
            assert max(ac_np_state_split) == features.shape[0], '{}!={}'.format(max(ac_np_state_split),
                                                                                features.shape[0])
            splited_feats = np.split(features, ac_np_state_split)
            for state_feat_list, feats in zip(self.feats_lists, splited_feats):
                state_feat_list.append(feats)
            self.state_transfer_historys.append(states)
            # print('states')
            # print(states)
            # print('split1')
            # print(np_state_split)
            # print('split2')
            # print(ac_np_state_split)
        # print(s1.shape)
        # print(s2.shape)
        # print(s3.shape)

    def init_gmm(self, verbose=False):
        """
        train gmm as init
        """
        res = []
        for i, feats_list in enumerate(self.feats_lists):
            feats = np.vstack(feats_list)
            if verbose:
                print('s{} data: {}'.format(i + 1, feats.shape))
            res.append(self.B[i].train(feats, kmeans_init=True, verbose=verbose))
        # After initialization, feats_lists is no longer needed
        return res

    def init_gmm2(self, verbose=False):
        """
        kmeans init gmm as init
        """
        for i, feats_list in enumerate(self.feats_lists):
            feats = np.vstack(feats_list)
            if verbose:
                print('s{} data: {}'.format(i + 1, feats.shape))
            self.B[i].kmeans_init(feats, verbose=verbose)
        # After initialization, feats_lists is no longer needed

    def update(self, verbose=False):
        # after assign feats to each state and get state transfer history
        # we can now do m-step
        # self.feats_lists, 3 list for each state
        # a group of assigned feats(shape=m*39) in each list
        # self.states_history, state transfer history
        feats = [np.vstack(feats_list) for feats_list in self.feats_lists]
        for gmm, feat in zip(self.B, feats):
            if verbose:
                print('start, feat:', feat.shape)
            gmm.reestimate(feat)
        N = self.N
        A = self.A
        count = np.zeros((N, N))
        for state_transfer_history in self.state_transfer_historys:
            for i in range(len(state_transfer_history) - 1):
                state_i = state_transfer_history[i]
                state_j = state_transfer_history[i + 1]
                count[state_i, state_j] += 1
        # add end
        count[N - 2, N - 1] += 1
        # smooth
        count[A != 0] += 1
        self.A = count / np.sum(count, axis=1, keepdims=True)

    def reset(self):
        self.feats_lists = [[] for _ in range(self.N - 1)]
        self.states_history = []

    def save(self, path):
        file = h5py.File(path, 'w')
        file.create_dataset('A', data=self.A)
        for i, B in enumerate(self.B):
            for j, (weight, mean, covar) in enumerate(zip(B.weights, B.means, B.covars)):
                file.create_dataset('B{}_weight{}'.format(i, j), data=weight)
                file.create_dataset('B{}_mean{}'.format(i, j), data=mean)
                file.create_dataset('B{}_covar{}'.format(i, j), data=covar)
        file.close()

    def load(self, path):
        file = h5py.File(path, 'r')
        self.A = file['A'][()]
        for i, B in enumerate(self.B):
            B.weights = [file['B{}_weight{}'.format(i, j)][()] for j in range(B.K)]
            B.means = [file['B{}_mean{}'.format(i, j)][()] for j in range(B.K)]
            B.covars = [file['B{}_covar{}'.format(i, j)][()] for j in range(B.K)]
        file.close()


class ConnectedCHMM:
    def __init__(self, hmms):
        self.hmms = hmms
        self.num_hmms = len(self.hmms)
        self.state_for_each_hmm = hmms[0].N
        self.n = self.state_for_each_hmm - 1  # state for each phoneme
        self.N = self.num_hmms * self.n + 1
        self.pi = [1] + [0 for _ in range(self.N - 1)]  # pi

        self.K = hmms[0].B[0].K
        self.wav_feats = None

        self.A = None
        self.B = None
        self.log_a = None
        self.log_b = None

        self._init()
        assert len(self.B) == self.N - 1

    def _init(self):
        A = np.zeros((self.N, self.N))
        B = []
        for i, hmm in enumerate(self.hmms):
            A[self.n * i:self.n * (i + 1), self.n * i:self.n * (i + 1) + 1] = hmm.A[:self.n, :self.n + 1]
            B += hmm.B
        A[-1, -1] = 1
        self.A = A
        self.B = B
        self.log_a = elog(self.A)

    def reset(self):
        # since A does dot share memory with hmm.A
        # we need to rebuild it after m-step
        A = np.zeros((self.N, self.N))
        for i, hmm in enumerate(self.hmms):
            A[self.n * i:self.n * (i + 1), self.n * i:self.n * (i + 1) + 1] = hmm.A[:self.n, :self.n + 1]
        A[-1, -1] = 1
        self.A = A
        self.log_a = elog(self.A)
        self.log_b = None

    def assign(self, wav_feats, states=None):
        # for viterbi training
        # if states is not given, average split features to each hmm
        self.wav_feats = wav_feats
        if states is None:  # average split
            for phoneme_hmm, phoneme_feats in zip(self.hmms, np.array_split(wav_feats, self.num_hmms)):
                phoneme_hmm.assign(phoneme_feats)
        else:
            assert len(states) - 1 == wav_feats.shape[0]
            states = states[:-1]  # remove end state
            phoneme_split = [state // self.n for state in states]
            phoneme_state_split = [state % self.n for state in states]
            assert max(phoneme_split) == len(self.hmms) - 1
            np_phoneme_split = [np.sum(np.array(phoneme_split) == i) for i in range(len(self.hmms))]
            assert sum(np_phoneme_split) == wav_feats.shape[0]

            ac_np_phoneme_split = np.add.accumulate(np_phoneme_split)
            assert max(ac_np_phoneme_split) == wav_feats.shape[0]
            splited_wav_feats = np.split(wav_feats, ac_np_phoneme_split)
            splited_states = np.split(np.array(phoneme_state_split), ac_np_phoneme_split)
            for phoneme_hmm, phoneme_feats, phoneme_states in zip(self.hmms, splited_wav_feats, splited_states):
                phoneme_hmm.assign(phoneme_feats, phoneme_states)

    def get_log_b(self, observation=None):
        if observation is not None:
            o = observation
            T = o.shape[0]
            b = elog(np.zeros((self.N, T + 1)))
            for i, B in enumerate(self.B):
                b[i, :T] = B.logpdf(o)
            assert len(self.B) == self.N - 1
            b[-1, -1] = 0
            return b
        if self.log_b is None:
            o = self.wav_feats
            T = o.shape[0]
            b = elog(np.zeros((self.N, T + 1)))
            for i, B in enumerate(self.B):
                b[i, :T] = B.logpdf(o)
            assert len(self.B) == self.N - 1
            b[-1, -1] = 0
            self.log_b = b
        return self.log_b

    def viterbi(self, obseravttion=None):
        if obseravttion is not None:
            o = obseravttion
        else:
            o = self.wav_feats
        T = o.shape[0]
        N = self.N

        log_b = self.get_log_b(obseravttion)

        log_a = self.log_a

        # log_delta0 = elog(np.zeros((N, T + 1)))
        # log_delta0[:, 0] = elog(np.array(self.pi)) + log_b[:, 0]

        log_delta1 = elog(np.zeros((N, T + 1)))
        log_delta1[:, 0] = elog(np.array(self.pi)) + log_b[:, 0]

        # for t in range(1, T + 1):
        #     for j in range(N):
        #         log_delta0[j, t] = log_b[j, t] + np.max(log_delta0[:, t - 1] + log_a[:, j])
        for t in range(1, T + 1):
            log_delta1[:, t] = log_b[:, t] + np.max(log_delta1[:, t - 1].reshape((-1, 1)) + log_a, axis=0)

        # assert np.allclose(log_delta0,log_delta1)
        try:
            assert not np.isnan(log_delta1).any()
        except AssertionError:
            print('cal process of log_delta')
            print('t=0')
            print('log pi:')
            print(elog(np.array(self.pi)))
            print('log b [:,0]')
            print(log_b[:, 0])
            flag = False
            for t in range(1, T + 1):
                for j in range(N):
                    print('t={},j={}'.format(t, j))
                    print('log_b[j,t]')
                    print(log_b[j, t])
                    print('log_delta[:,t-1]')
                    print(log_delta1[:, t - 1])
                    print('log_a[:,j]')
                    print(log_a[:, j])
                    print('log_delta[:,t-1]+log_a[:,j]')
                    print(log_delta1[:, t - 1] + log_a[:, j])
                    print('max')
                    print(np.max(log_delta1[:, t - 1] + log_a[:, j]))
                    print('argmax')
                    print(np.argmax(log_delta1[:, t - 1] + log_a[:, j]))
                    if np.isnan(log_delta1[j, t]).any():
                        flag = True
                        break
                if flag:
                    break
            raise

        return log_delta1

    def decode_viterbi(self, observation=None):
        log_delta = self.viterbi(observation)
        T = log_delta.shape[1]
        N = self.N

        states = -1 * np.ones((T,), dtype=int)
        states[-1] = np.argmax(log_delta[:, -1])
        for t in range(T - 2, -1, -1):
            if states[t + 1] == N - 1:
                states[t] = N - 2
            elif states[t + 1] == 0:
                states[t] = 0
            else:
                states[t] = states[t + 1] if log_delta[states[t + 1], t] > log_delta[states[t + 1] - 1, t] else states[
                                                                                                                    t + 1] - 1

        try:
            self.check_states(states,observation)
        except AssertionError:
            np.set_printoptions(formatter={'float': '{: 0.2e}'.format}, threshold=1000000)
            print('state error')
            print('A')
            for i in range(self.A.shape[0]):
                print('A {}: {}'.format(i, self.A[i]))
            print('b')
            logb = self.get_log_b()
            for i in range(logb.shape[0]):
                print('logb {}: {}'.format(i, logb[i]))
            print('log_delta')
            for i in range(log_delta.shape[0]):
                print('log_delta {}: {}'.format(i, log_delta[i]))
            print('state')
            print(states)
            raise

        return states, log_delta[-1, -1]

    def check_states(self, states,observation=None):
        assert states[0] == 0
        assert states[-1] == self.N - 1
        diff_states = np.diff(states)
        assert (diff_states[diff_states != 0] == 1).all()
        if observation is None:
            observation = self.wav_feats
        assert len(states) == observation.shape[0] + 1

    def forward(self, observation=None):
        if observation is None:
            o = self.wav_feats
        else:
            o = observation
        T = o.shape[0]
        n = o.shape[1]
        N = self.N
        assert o is not None
        log_a = self.log_a
        log_b = self.get_log_b(observation)

        # log_alpha0 = np.zeros((N, T + 1))
        # log_alpha0[:, 0] = elog(np.array(self.pi)) + log_b[:, 0]

        log_alpha1 = np.zeros((N, T + 1))
        log_alpha1[:, 0] = elog(np.array(self.pi)) + log_b[:, 0]

        # for t in range(1, T + 1):
        #     for i in range(N):
        #         log_alpha0[i, t] = logsumexp(log_a[:, i] + log_alpha0[:, t - 1]) + log_b[i, t]
        for t in range(1, T + 1):
            log_alpha1[:, t] = logsumexp(log_a + log_alpha1[:, t - 1].reshape((-1, 1)), axis=0) + log_b[:, t]
        # assert np.allclose(log_alpha0, log_alpha1)
        return logsumexp(log_alpha1[:, -1]), log_alpha1

    def forward_backward(self):
        # to avoid cal b
        o = self.wav_feats
        T = o.shape[0]
        n = o.shape[1]
        assert o is not None

        b = self.get_b()
        log_a = elog(self.A)
        log_b = elog(b)
        # print('log_b')
        # print(log_b.shape)
        # print(log_b)

        # foward
        log_alpha = np.zeros((self.N, T + 1))
        log_alpha[:, 0] = elog(np.array(self.pi)) + log_b[:, 0]
        # print('elog pi')
        # print(elog(np.array(self.pi)))
        # print('log alpha 0')
        # print(log_alpha[:, 0])
        for t in range(1, T + 1):
            # TODO optimize to one loop
            for i in range(self.N):
                log_alpha[i, t] = logsumexp(log_a[:, i] + log_alpha[:, t - 1]) + log_b[i, t]
            # alpha[:, t] = self.A.T.dot(alpha[:, t - 1]) * b[:, t]

        # backward
        log_beta = np.zeros((self.N, T + 1))
        for t in range(T - 1, -1, -1):
            for i in range(self.N):
                log_beta[i, t] = logsumexp(log_a[i, :] + log_beta[:, t + 1] + log_b[:, t + 1])

        return log_alpha, log_beta


if __name__ == '__main__':
    # gmms = [GMM(2, 2), GMM(2, 2), GMM(2, 2)]
    # hmm = CHMM(3, None, np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]]),
    #            gmms, [1, 0, 0])
    # sh, oh = hmm.sample(5)
    # print('state:', sh)
    # print('observed:', oh)
    np.random.seed(111)
    hmm_init = {
        'N': 4,
        'A': [[0.5, 0.5, 0, 0],
              [0, 0.5, 0.5, 0],
              [0, 0, 0.5, 0.5],
              [0, 0, 0, 1]],
        'GMM_kwargs': {'K': 3, 'n': 39, 'covar_type': 'diag'},
        'pi': [1, 0, 0, 0]
    }
    o = np.random.randn(100, 39)
    hmm1 = CHMM(**hmm_init)
    hmm2 = CHMM(**hmm_init)
    # hmm1.load('./hmm1.hdf5')
    # hmm2.load('./hmm2.hdf5')
    joinhmm = ConnectedCHMM([hmm1, hmm2])
    res = joinhmm.forward(o)
    print(res)
