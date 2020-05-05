import numpy as np


def multinomial_sample(probability):
    return np.random.multinomial(1, probability, 1).argmax()


def logSumExp(x, axis=None, keepdims=False, do_elog=False):
    max_x = np.max(x,axis=axis,keepdims=True)
    diff_x = x - max_x
    sum_exp = np.exp(diff_x).sum(axis=axis, keepdims=keepdims)
    # print('sum exp')
    # print(sum_exp)
    print(max_x.shape)
    if not keepdims:
        max_x = np.squeeze(max_x, axis=axis)
    print(sum_exp.shape)
    print(max_x.shape)
    if do_elog:
        return max_x + elog(sum_exp)
    else:
        return max_x + np.log(sum_exp)


def elog(x):
    res = np.log(x, where=(x != 0))
    res[np.where(x == 0)] = -(10.0 ** 8)
    return res
