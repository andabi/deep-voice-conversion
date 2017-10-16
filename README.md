# Voice Conversion

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.1
  * librosa

## Data

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `data_load.py` loads data and put them in queues so multiple mini-bach data are generated in parallel.
  * `utils.py` has several custom operational functions.
  * `modules.py` contains building blocks for encoding/decoding networks.
  * `train.py` is for training.
  * `eval.py` is for sample synthesis.  

## Training

## Sample Synthesis
  * Run `eval.py` to get samples.

### Acknowledgements
