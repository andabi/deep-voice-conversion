# Voice Conversion with Non-Parallel Data
> Special thanks to [Kyubyong Park](https://github.com/Kyubyong). (Co-worker)
## Intro

What if you could imitate a famous celebrity's voice or sing like a famous singer?
This project started with a goal to convert anyone's voice to a specific target voice.
So called, it's voice style transfer. (or Voice-to-Voice)
I implemented a deep neural networks to achieve that.
I could convert a anonymous male's voice to a famous English actress [Kate Winslet](https://en.wikipedia.org/wiki/Kate_Winslet)'s voice.
Don't hesitate to visit [here](https://soundcloud.com/andabi/sets/voice-style-transfer-to-kate-winslet-with-deep-neural-networks) to listen those!

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
* window length: 25ms
* hop length: 5ms

### Datasets
* Train1
  * [TIMIT dataset](https://catalog.ldc.upenn.edu/ldc93s1)
    * contains 630 speakers' utterances and corresponding phones.
* Train2
  * Target1(anonymous female): [Arctic](http://www.festvox.org/cmu_arctic/) dataset (public)
  * Target2(Kate Winslet): Audiobook read by Kate (private)
    * Splitted by sentences.

### Samples
[here](https://soundcloud.com/andabi/sets/voice-style-transfer-to-kate-winslet-with-deep-neural-networks)

## Tips (Lessons I've learned from this project)
* Window length and hop length should be small enough to fit in only one phoneme.
* Obviously, sample rate, window length and hop length should be same in both Net1 and Net2.
* It seems that the accuracy of Net1(phoneme classification) does not need to be perfect.
  * Net2 can reach to near optimal when Net1 accuracy is correct to some extent.
* Before ISTFT(spectrogram to waveforms), emphasizing on the predicted spectrogram by applying power of 1.0~2.0 is helpful for removing noisy sound.
* It seems that to apply temperature to softmax in Net1 is meaningless.

## Future Works
* Adversarial training
  * Expecting to generate sharper and cleaner voice.
* Cross lingual

## Ultimate Goals
* Many-to-Many(Multi target speaker) voice conversion system
* VC without training set of target voice, but only small set of target voice (1 min)
  * (On going)

## References
* ["Phonetic posteriorgrams for many-to-one voice conversion without parallel data training"](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training), 2016 IEEE International Conference on Multimedia and Expo (ICME)
* ["TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS"](https://arxiv.org/abs/1703.10135), Submitted to Interspeech 2017