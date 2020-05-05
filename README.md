# Command Word Recognition
Command Word Recognition is based on a phoneme level GMM-HMM embedding model
## Steps
You skip first 3 steps and do step4 to test my reuslts. 
### Step1, Build lexicon
Build lexicon which maps a character to phonemes
```angular2
python DaCiDian/DaCiDian.py
```
### Step2, Prepare dataset 
```angular2
python prep_data.py --mfcc_dim 13 --min_thresh 5
```
phoneme whose frequency less than 5 will be replaced by 'UNK'
### Step3, Start training
```angular2
python main.py
```
it takes about 4 min for a step. Default GMM feature dim is 13. It can
give better acc on test set than dim=39
### Step4, Test
```angular2
python test.py
```
Remember to change model_path in test.py. It will give viterbi align and accuracy.
## Results
results of 39 dim mfcc feature:

![](https://github.com/luo3300612/command_word_rec/blob/master/assets/accuracy.png)

![](https://github.com/luo3300612/command_word_rec/blob/master/assets/loglikelihood.png)

results of 13 dim mfcc feature:

![](https://github.com/luo3300612/command_word_rec/blob/master/assets/accuracy39.png)

![](https://github.com/luo3300612/command_word_rec/blob/master/assets/loglikelihood39.png)

viterbi align:

![](https://github.com/luo3300612/command_word_rec/blob/master/assets/align.png)

## References
* [DaCiDian](https://github.com/aishell-foundation/DaCiDian)
* [hmmlearn](https://github.com/hmmlearn/hmmlearn)
* https://zhuanlan.zhihu.com/p/55826713
* https://github.com/desh2608/gmm-hmm-asr/blob/master/submission.py
* https://blog.csdn.net/chinatelecom08/article/details/82901480
