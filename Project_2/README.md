# Project 2

## Topic
Image classification with convolutional neural networks

## Plan

- [X] Loading and exploring dataset
- [X] Training simple, own written CNN ~58%
- [X] Training simple CNN with simple data augmentation
- [ ] Training simple CNN with more sophisticated data augmentation (for example AutoAugment, cutout)
- [ ] Training simple CNN with different optimizer, learning rate, and step size
- [X] Training CNN with pretrained ResNet weights ~ 92%
- [ ] Training CNN with pretrained ResNet weights with different data augmentation, optimizer, etc.
- [ ] Training CNN with other pretrained models
- [ ] Implementation of TTA
- [ ] Ensembling of few previously trained models
- [ ] ...

## Datasets

CIFAR-10 (https://www.kaggle.com/c/cifar-10/)

## Requirements

* utilizing code from external sources - only with references and applying some modifications
* application of pretrained models (like AlexNet, VGG) - permitted, recommended

* reproducibility (random number generator)
* each experiment should be repeated multiple times

## Project conspect:

* problem description
* reaserch goals
* planned methods along with references
* data description

## Ideas

* basic CNN -> increasing number of layer/size of layers
* data augmentation
* test time augmentation
* ensembling (soft voting or majority/hard voting)
* using models pretrained on ImageNet (for example EfficientNet)

## Timetable

* 30.03 - Verification of project plan
* 13.04 - Initial presentation of the Project 2
* 20.04 - Project 2 deadline

## Useful resources

* Google Colab or GPU to speed up training
* ensembling (soft voting or majority/hard voting)
* data augmentation
* https://benchmarks.ai/cifar-10
* https://www.robots.ox.ac.uk/vgg/practicals/cnn/index.html
* https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner’s-Guide-To-Understanding-Convolutional-Neural-Networks/
* https://medium.com/kaggle-blog/profiling-top-kagglers-bestfitting-currently-1-in-the-world-58cc0e187b
