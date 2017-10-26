# Voice Conversion
> With special thanks to a amazing AI researcher [Kyubyong Park](https://github.com/Kyubyong). (Co-worker)
## Intro

What if you could imitate a famous celebrity's voice or sing like a famous singer?
This project started with a goal to convert anyone's voice to a specific target voice.
So called, it's voice style transfer. (or Voice-to-Voice)
I implemented a deep neural networks to achieve that.
I could convert a anonymous male's voice to a famous English actress [Kate Winslet](https://en.wikipedia.org/wiki/Kate_Winslet)'s voice.
Don't hesitate to visit [here]() to listen those!

## Model Architecture
This is a many-to-one voice conversion system.
The main significance of this work is that we could generate a target speaker's utterances without parallel datasets like <wav, text> or <wav, phone>, but only waveforms of the target speaker.
(To get or create a set of <wav, phone> pairs from a target speaker needs a lot of effort.)
All we need is a number of waveforms of the target speaker's utterances and only a small set of <wav, phone> pairs from anonymous speakers.

The model architecture consists of two modules:
1. phoneme recognition: classify anyone's utterances to one of phoneme classes (Net1)
2. speech generation: synthesize speech of target speaker from the phones. (Net2)

Simply, I applied CBHG(1-D convolution bank + highway network + bidirectional GRU) modules that are mentioned in [Tacotron](https://arxiv.org/abs/1703.10135) paper.

overview picture

### Net1 is a classifier.
* Process: wav -> spectrogram -> mfccs -> ppgs
* Net1 classifies Spectrogram to phonemes that consists of 60 English phonemes along timesteps.
  * The input is a spectrogram of log of linear magnitude.
  * The output is a probability distribution of phonemes named PPG(phonetic posteriorgrams) for each timestep.
* Training
  * Cross entropy loss.
* Around 70% accuracy achieved in evaluation
### Net2 is a generator.
Net2 contains Net1 as a sub-network.
* Process: net1(wav -> spectrogram -> mfccs -> ppgs) -> spectrogram -> wav
* Net2 synthesizes the target speaker's speeches.
  * The input is a set of target speaker's utterances.
* Training
  * Since it's assumed that Net1 is already trained in previous step, only the remaining part should be trained in this step.
  * Loss is reconstruction error
    * L2 distance between spectrogram before/after Net2.
* Griffin-Lim reconstruction is used in process of reverting wav from spectrogram.
## Implementations
### Procedure
* Train phase: training Net1 and Net2 sequentially.
  * Train1: training Net1
    * Run `train1.py` to train.
    * Run `eval1.py` to test.
  * Train2: training Net2
    * Run `train2.py` to train.
      * Warning: should be trained after train1 is done!
    * Run `eval2.py` to test.
* Convert phase: feed forwarding in Net2
    * Run `convert.py` to get samples.
    * Check Tensorboard's audio tab to listen the samples.
    * Check the visualized phoneme distributions(ppgs) along timestep in Tensorboard's image tab.
      * x-axis: phoneme classes
      * y-axis: timestep

### Settings
* sample rate: 16,000Hz
* frame length: 25ms
* frame shift: 5ms

### Datasets
* Train1
  * [TIMIT dataset](https://catalog.ldc.upenn.edu/ldc93s1)
    * contains 630 speakers' utterances and corresponding phones.
* Train2
  * Target1(anonymous female): [Arctic](http://www.festvox.org/cmu_arctic/) dataset (public)
  * Target2(Kate Winslet): Audiobook read by Kate (private)
    * Splitted by sentences.

### Samples
[here]()

## Tips
* phoneme classification 할땐 window_len, hop_len가 하나의 class로 매핑될 수 있게 충분히 작아야 한다. 각각이 25ms, 5ms인 이유.
* 각 모듈(net1, net2)에서 쓰는 sample rate, window_len, hop_len는 같아야 한다.
* train1의 classification 정확도는 완벽할 필요는 없음.
 - training set의 label이 완벽하지 않아도 (수렴하는데까지는 더 오래 걸리지만) 결국 완벽할 때의 optimal에 도달한다는 hinton의 연구 결과 참고
* spectrogram -> wav 변환시 magnitude에 어느 정도 emphasis를 주는 것은 잡음을 제거하는데 도움이 된다.
* softmax temperature로 ppg dist.를 임의로 바꾸는 것은 크게 도움되지 않는다.
 - 어차피 class의 수는 동일하고 0~1 사이의 범위라고는 하지만 실수 범위이기 때문에 담을 수 있는 정보량은 동일.
 - cf) autoencoder vs variational autoencoder

## Future Works
* Adversarial training
  * Expecting to generate sharper and cleaner voice.


## Acknowledgements
This work mainly refers to a paper ["Phonetic posteriorgrams for many-to-one voice conversion without parallel data training"](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training), 2016 IEEE International Conference on Multimedia and Expo (ICME)