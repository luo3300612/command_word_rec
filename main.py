import json
import time

from tqdm import tqdm
import numpy as np
from python_speech_features import *
import scipy.io.wavfile as wav
from gmm import GMM
from gmm_hmm import CHMM, ConnectedCHMM, logSumExp
import h5py
from utils import elog
from collections import Counter
from datetime import datetime
import os

covar_type = 'diag'

hmm_init = {
    'N': 4,
    'A': [[0.5, 0.5, 0, 0],
          [0, 0.5, 0.5, 0],
          [0, 0, 0.5, 0.5],
          [0, 0, 0, 1]],
    'GMM_kwargs': {'K': 3, 'n': 13, 'covar_type': covar_type},
    'pi': [1, 0, 0, 0]
}
model_save_path = './saved_models'
np.random.seed(1234)

if __name__ == '__main__':
    # read dataset
    dataset = json.load(open('./dataset.json', encoding='utf8'))
    features = h5py.File('./features.hdf5', 'r')
    phonemes_set = dataset['phonemes']
    print('Phonemes Classes: {}'.format(len(phonemes_set)))

    # make save dir
    save_path = os.path.join(model_save_path, datetime.now().strftime('%m%d_%H%M_%S'))
    os.mkdir(save_path)

    # assign phoneme hmm
    phoneme_hmms = {}
    for phoneme in phonemes_set:
        phoneme_hmms[phoneme] = CHMM(**hmm_init)

    # k-means initialization
    # split wav and assgin feats
    wav_hmm = {}
    for entry in dataset['wavs']:
        wav_id = entry['id']
        wav_phonemes = entry['phonemes']['formatted']
        wav_features = features[wav_id]

        phonemes_len = len(wav_phonemes)

        wav_hmm[wav_id] = ConnectedCHMM([phoneme_hmms[wav_phoneme] for wav_phoneme in wav_phonemes])
        wav_hmm[wav_id].assign(wav_features)

    print('k-menas init...')
    c = Counter()
    for phoneme in tqdm(phonemes_set):
        res = phoneme_hmms[phoneme].init_gmm2(verbose=False)
        phoneme_hmms[phoneme].reset()

    print('init OK')
    eps = 1
    max_steps = 20
    last_dataset_loglikelihood = 0
    dataset_loglikelihood_history = np.zeros((max_steps,))
    for step in range(max_steps):
        print('save model...')
        step_save_path = os.path.join(save_path, 'step{}'.format(step + 1))
        os.mkdir(step_save_path)
        for phoneme, phoneme_hmm in phoneme_hmms.items():
            phoneme_hmm.save(os.path.join(step_save_path, phoneme + '.hdf5'))
        np.save(os.path.join(save_path, 'history.npy'), dataset_loglikelihood_history)

        # e-step
        dataset_loglikelihood = 0
        with tqdm(desc='Epoch {}, E-step'.format(step+1), unit='it', total=len(wav_hmm)) as pbar:
            for it, (wav_id, hmm) in enumerate(wav_hmm.items()):
                hmm.reset()
                wav_features = features[wav_id]
                states, logprob = hmm.decode_viterbi()
                loglikelihood, log_alpha = hmm.forward()
                dataset_loglikelihood += loglikelihood
                hmm.assign(wav_features, states)
                pbar.set_postfix(loglikelihood=loglikelihood, len=hmm.N)
                pbar.update()

        dataset_loglikelihood = dataset_loglikelihood / len(wav_hmm)
        dataset_loglikelihood_history[step] = dataset_loglikelihood
        print('Average Loglikelihood:', dataset_loglikelihood)
        if np.fabs(dataset_loglikelihood - last_dataset_loglikelihood) < eps:
            print('eps reached, Done')
            break
        last_dataset_loglikelihood = dataset_loglikelihood
        # m-step
        with tqdm(desc='Epoch {}, M-step'.format(step+1), unit='it', total=len(phoneme_hmms)) as pbar:
            for it, (phoneme, phoneme_hmm) in enumerate(phoneme_hmms.items()):
                phoneme_hmm.update(verbose=False)
                phoneme_hmm.reset()

                pbar.set_postfix(phoneme=phoneme)
                pbar.update()
    else:
        print('step reached, Done')
    print('Done')
