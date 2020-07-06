# Command Word Recognition
Command Word Recognition is based on a phoneme level GMM-HMM embedding model
## dependency
```angular2
pip install -r requirements.txt
```
## Steps
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
it takes about 4 min for a step on my Win10 and about 45s on my Manjaro Linux. Default GMM feature dim is 13. It can
give better acc on test set than dim=39
### Step4, Test
```angular2
python test.py
```
Remember to change model_path in test.py. It will give viterbi align and accuracy.
## Results
results of 13 dim mfcc feature:

![](https://github.com/luo3300612/command_word_rec/raw/master/assets/loglikelihood.png)

![](https://github.com/luo3300612/command_word_rec/raw/master/assets/accuracy.png)


results of 39 dim mfcc feature:

![](https://github.com/luo3300612/command_word_rec/raw/master/assets/loglikelihood39.png)

![](https://github.com/luo3300612/command_word_rec/blob/master/assets/accuracy39.png?raw=true)

viterbi align example:

![](https://github.com/luo3300612/command_word_rec/raw/master/assets/align.png)

## References
* [DaCiDian](https://github.com/aishell-foundation/DaCiDian)
* [hmmlearn](https://github.com/hmmlearn/hmmlearn)
* https://zhuanlan.zhihu.com/p/55826713
* https://github.com/desh2608/gmm-hmm-asr/blob/master/submission.py
* https://blog.csdn.net/chinatelecom08/article/details/82901480
