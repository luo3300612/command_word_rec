import os
import json
import h5py
import argparse
from collections import Counter

from pathlib import Path
import numpy as np
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# About Data
data_root = 'data'
phone_numbers = ['13776677136']
datasets = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# About prepare
punctuations = ['，', '。', '、', '？', '！', '“', '”', '：', '《', '》', '?', '.', '—', ',', '…']
reference = json.load(open('./DaCiDian/lexicon.json'))
sp_punctuations = ['，', '。', '、', '？', '！', '：', '?', '.', '—', ',', '…']
other_punctuations = set(punctuations) - set(sp_punctuations)


def word2phoneme(word):
    """
    transfer a word to phonemes
    """
    return 'sp' if word in sp_punctuations else reference.get(word, False)


def get_mfcc(filename, return_wav=False, dim=13):
    (fs, data) = wav.read(filename)
    wav_feature = mfcc(data, fs)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    feature = feature.astype('float32')
    # print(feature.dtype)
    if dim == 13:
        feature = wav_feature.astype('float32')
    if return_wav:
        return feature, data
    return feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_thresh', type=int, default=5, help='min freq of phoneme')
    parser.add_argument('--mfcc_dim', type=int, default=13)
    args = parser.parse_args()

    out = []
    features = h5py.File('features.hdf5', 'w')
    for dataset in datasets:
        # build sent_id : sent map
        with open(os.path.join(data_root, 'notes', dataset + '.txt'), encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            formatted = {line.split('、')[0]: '、'.join(line.split('、')[1:]) for line in lines}

        # sent to phonemes
        # build sent_id : sent_info map
        id2infos = {}
        for key, value in formatted.items():
            words = list(value)
            # sentence to phonemes, default add sp when encounter sp_punctuations
            token_phonemes = [word2phoneme(word) for word in words if word not in other_punctuations]
            if token_phonemes[-1] == 'sp':
                del token_phonemes[-1]  # since sil is inserted, we do not need sp any more
            assert False not in token_phonemes
            phonemes = ['sil']  # add sil before
            for token_phoneme in token_phonemes:
                phonemes += token_phoneme.split(' ')
            phonemes.append('sil')  # add sil after
            id2infos[key] = {'phonemes': phonemes, 'raw': value}
            # print('文本:', value)
            # print('音素序列:', phonemes)

        # build dataset : feature and infos
        # we only need part of each note file
        # because only part of sentences are recorded
        for phone_number in phone_numbers:
            wavpath = os.path.join(data_root, phone_number, dataset, 'm_' + phone_number)
            path = Path(wavpath)
            for wav_file in path.iterdir():
                filename = wav_file.name
                wav_id = filename.split('.')[0].split('_')[1]
                infos = id2infos[wav_id]
                entry = {'wav_path': str(wav_file), 'raw': infos['raw'], 'phonemes': infos['phonemes'],
                         'id': filename.split('.')[0]}
                out.append(entry)

                features.create_dataset(entry['id'], data=get_mfcc(entry['wav_path'],dim=args.mfcc_dim))

    # statistic
    c = Counter()
    for entry in out:
        wav_phonemes = entry['phonemes']
        c.update(wav_phonemes)

    # insert UNK
    for entry in out:
        wav_phonemes = entry['phonemes']
        formatted_phonemes = [phoneme if c[phoneme] >= args.min_thresh else 'UNK' for phoneme in wav_phonemes]
        entry['phonemes'] = {'raw': wav_phonemes, 'formatted': formatted_phonemes}

    # statistic
    num_phonemes = sum([v for _, v in c.items()])
    sorted_list = sorted([(k, v) for k, v in c.items()], key=lambda x: x[1], reverse=True)
    removed_phonemes = []
    total_freq = 0
    print('phoneme frequent')
    for k, v in sorted_list:
        print('{}: {:.4f}%, {}/{}'.format(k, v / num_phonemes * 100, v, num_phonemes))
        if v < args.min_thresh:
            removed_phonemes.append(k)
            total_freq += v
    print('Phonemes Classes:', len(c))
    print('Phonemes Num:', num_phonemes)
    print('removed phonemes:', removed_phonemes)
    print('removed phonemes percentage: {}/{} ({:.4f}%) {}/{}({:.4f}%)'.format(len(removed_phonemes), len(c),
                                                                               len(removed_phonemes) / len(c) * 100,
                                                                               total_freq, num_phonemes,
                                                                               total_freq / num_phonemes * 100))
    phonemes = list(set(c.keys()) - set(removed_phonemes))
    if len(removed_phonemes) > 0:
        phonemes.append('UNK')
    print('Phonemes after UNK inserted: ', len(phonemes))

    out = {'wavs': out, 'phonemes': phonemes}
    features.close()
    json.dump(out, open('dataset.json', 'w', encoding='utf8'))
    print('dataset size:', len(out['wavs']))
    print('infos saved to dataset.json')
    print('features saved to feature.hdf5')
    print('Done')

    # print('前进')
    # print(c['q'])
    # print(c['ian_2'])
    # print(c['j'])
    # print(c['in_4'])
    # print('后退')
    # print(c['h'])
    # print(c['ou_4'])
    # print(c['t'])
    # print(c['ui_4'])
    # print('左转')
    # print(c['z'])
    # print(c['uo_3'])
    # print(c['zh'])
    # print(c['uan_3'])
    # print('右转')
    # print(c['y'])
    # print(c['ou_4'])
    # print(c['zh'])
    # print(c['uan_3'])
    # sorted_c = sorted([(key, value) for key, value in c.items()], key=lambda x: x[1], reverse=True)
    # plt.bar(np.arange(len(c)), list(map(lambda x: x[1], sorted_c)))
    # plt.title('phoneme frequency')
    # # plt.xticks(np.arange(len(c)), list(map(lambda x: x[0], sorted_c)),rotation='vertical')
    # plt.show()
