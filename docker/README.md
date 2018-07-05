# Training with Nvidia-Docker
## Prerequisites
Because of dependencies of the GPU (cuda etc), there are 3 steps if you're starting from scratch on a 16.04 machine
- install cuda
- install docker
- install nvidia-docker

## Quickstart
first build the image
```
nvidia-docker build -t $USER/deep-voice-transfer-gpu:latest .
```

then run
```
nvidia-docker run -it --rm $USER/deep-voice-transfer-gpu:latest
```

## Got-chas
- Notice the pip-requirement does not have tensorflow, but instead tensorflow-gpu. I didn't quite figure out how to force it to use GPU (even with the `-gpu 0` flag) so having tensorflow also installed meant that CPU is used so you "hang" at 0% when you train weights for net2
