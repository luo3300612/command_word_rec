#!/usr/bin/env python

# This script processes DaCiDian
# FROM:
# 	layer-1 mapping: DaCiDian/word_to_pinyin.txt
# 	layer-2 mapping: DaCiDian/pinyin_to_phone.txt
# TO: 
# 	lexicon.txt

import sys
import json

out = {}

syllable_to_phones = {}

word_to_syllable_file = 'DaCiDian/word_to_pinyin.txt'  # layer-1 mapping
syllable_to_phone_file = 'DaCiDian/pinyin_to_phone.txt'  # layer-2 mapping

for l in open(syllable_to_phone_file, encoding='utf-8'):  # "ZHENG	zh eng"
    cols = l.strip().split('\t')
    assert (len(cols) == 2)
    syllable = cols[0]
    phones = cols[1].split()
    syllable_to_phones[syllable] = phones

for l in open(word_to_syllable_file, encoding='utf-8'):  # "15	YI_1 WU_3;YAO_1 WU_3"
    cols = l.strip().split('\t')
    assert (len(cols) == 2)
    word = cols[0]
    prons = cols[1].split(';')
    for pron in prons:
        phone_seq = []
        for syllable in pron.split():
            base, tone = syllable.split('_')
            phones = [phn for phn in syllable_to_phones[base]]
            phones[-1] = phones[-1] + '_' + tone
            phone_seq.extend(phones)
        out[word] = ' '.join(phone_seq)
    # sys.stdout.write(word + '\t' + ' '.join(phone_seq) + '\n')

json.dump(out, open('DaCiDian/lexicon.json', 'w', encoding='utf-8'))
print('Done')
