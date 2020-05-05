from hmm_gmm import CHMM, ConnectedCHMM
import numpy as np
from main import covar_type, hmm_init
import json
import os
import h5py
from tqdm import tqdm
from prep_data import word2phoneme, get_mfcc
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

command_words = ['前进', '后退', '左转', '右转']
model_path = './saved_models/0503_1644/step11'

# change to True if you want to see viterbi align
show = True


def show_align(data, states, phonemes):
    """
    visualize viterbi align
    """
    n_windows = len(states)
    phoneme_align = np.array([state // 3 for state in states])
    ratio = round(len(data) / n_windows)

    locations = [sum(phoneme_align == i) for i in range(len(phonemes))]
    locations = [location * ratio for location in locations]
    locations = np.add.accumulate(locations)
    locations = locations.tolist()
    locations.insert(0, 0)
    locations = locations[:-1]

    plt.plot(data)
    plt.title('viterbi align')
    plt.xticks(locations, phonemes, rotation=60)
    plt.show()


if __name__ == '__main__':
    dataset = json.load(open('./dataset.json', encoding='utf8'))
    phonemes_set = dataset['phonemes']
    print('Phonemes Classes: {}'.format(len(phonemes_set)))

    # build phoneme hmms and load model
    phoneme_hmms = {}
    for phoneme in phonemes_set:
        phoneme_hmms[phoneme] = CHMM(**hmm_init)
        phoneme_hmms[phoneme].load(os.path.join(model_path, phoneme + '.hdf5'))

    # connect phoneme hmm to command word hmm
    command_words_phonemes = {}
    command_words_hmms = {}
    for command_word in command_words:
        words_phonemes = []
        for word in list(command_word):
            words_phonemes += word2phoneme(word).split(' ')
        words_phonemes.insert(0, 'sil')
        words_phonemes.append('sil')
        print(words_phonemes)
        command_words_phonemes[command_word] = words_phonemes
        for phoneme in words_phonemes:
            command_words_hmms[command_word] = ConnectedCHMM([phoneme_hmms[phoneme] for phoneme in words_phonemes])

    # start test
    acc = 0
    for command_word in command_words:
        for i in range(1, 6):
            wav_path = os.path.join('./data/test_data', '{}{}.wav'.format(command_word, i))
            wav_feat, data = get_mfcc(wav_path, return_wav=True)
            print('feat shape:', wav_feat.shape)
            res = []
            # test core
            for cw, hmm in command_words_hmms.items():
                states, logprob = hmm.decode_viterbi(wav_feat)
                loglikelihood, log_alpha = hmm.forward(wav_feat)
                res.append((cw, loglikelihood))
                if cw == command_word and show:
                    show_align(data, states, command_words_phonemes[command_word])
            # sort loglikelihood
            sorted_res = sorted(res, key=lambda x: x[1], reverse=True)

            if sorted_res[0][0] == command_word:
                acc += 1
                print('{}{} recog res:\n{} √'.format(command_word, i, res))
            else:
                print('{}{} recog res:\n{} X'.format(command_word, i, res))

    print('ACC: {}/{}, {:.2f}%'.format(acc, 20, acc / 20 * 100))
