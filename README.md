# Voice Conversion
> Co-worker: [Kyubyong Park](https://github.com/Kyubyong)
## Intro

What if you could imitate a famous celebrity's voice or sing like a famous singer?
This project started with a goal that is to convert anyone's voice to a specific target voice.
So called, it's voice style transfer or voice2voice.
I implemented a model based on deep neural networks to do that.
I can't say it's perfect but I could convert a anonymous man's voice to a famous English actress [Kate Winslet](https://en.wikipedia.org/wiki/Kate_Winslet)'s voice.
Just visit [here]() to listen those!

## Model Architecture
This is many to one voice conversion system.
The main significance of this work is that we could generate a target speaker's utterances without pairs of <wav, text> or <wav, phone>, but only waveforms of the target speaker.
To get or create a set of <wav, phone> pairs from a target speaker needs a lot of effort.
All we need is a number of wav of target speaker's utterances and a small set of <wav, phone> pairs from anonymous speakers.

The model architecture consists of two modules:
1. phoneme recognition: anyone's utterances to phone distributions. (net1)
2. voice generation of target speaker: speech synthesis from the phone distributions. (net2)

Simply, I applied CBHG(1-D convolution bank + highway network + bidirectional GRU) modules that are mentioned in [Tacotron](https://arxiv.org/abs/1703.10135) paper.

overview picture

### Settings
* sample rate: 16,000Hz
* frame length: 25ms
* frame shift: 5ms

### Net1 is a classifier.
* Process: wav -> spectrogram -> mfccs -> ppgs
* Net1 classifies Spectrogram to phonemes that consists of 60 English phonemes along timesteps.
  * The input is a spectrogram of log of linear magnitude.
  * The output is a probability distribution of phonemes named PPG(phonetic posteriorgrams) for each timestep.
* Training
  * [TIMIT dataset](https://catalog.ldc.upenn.edu/ldc93s1) which contains 630 speakers' utterances and corresponding phones is used for training/evaluation.
  * Cross entropy loss is used.
* Around 70% accuracy achieved in evaluation
### Net2 is a generator.
Net2 contains Net1 as a sub-network.
* Process: net1(wav -> spectrogram -> mfccs -> ppgs) -> spectrogram -> wav
* Net2 synthesizes the target speaker's speeches.
  * The input is a set of target speaker's utterances.
* Training
  * Since it's assumed that Net1 is already trained in previous step, only the remaining part should be trained in this step.
  * Datasets
    * Target1: [Arctic](http://www.festvox.org/cmu_arctic/) dataset (public)
    * Target2: Audiobook read by Kate (private)
      * Splitted by sentences
  * Loss is reconstruction error
    * L2 distance between spectrogram before/after Net2.
* Griffin-Lim reconstruction is used in process of reverting wav from spectrogram.
## Implementations
### Process
* Training phase: training Net1 and Net2 sequentially.
  * Train1: training Net1
    * Run `train1.py` to train.
    * Run `eval1.py` to test.
  * Train2: training Net2
    * Run `train2.py` to train.
      * Warning: should be trained after train1 is done!
    * Run `eval2.py` to test.
* Converting phase: feed forwarding in Net2
    * Run `convert.py` to get samples.
    * Check Tensorboard's audio tab to listen the samples.

## Future Works
* Adversarial training
  * Expecting to generate sharper and cleaner voice.


## Acknowledgements
This work mainly refers to a paper ["Phonetic posteriorgrams for many-to-one voice conversion without parallel data training"](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training), 2016 IEEE International Conference on Multimedia and Expo (ICME)