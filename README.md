# SVHN Image and Label Synthesis 
_3rd Place at EESTec AI Marathon, ETH Zurich 2024_

![image](https://github.com/radimurban/svhn-synthesis/assets/78273894/5a237304-59d2-42f4-900a-4ca3c4aaa099)

We implement a conditional Deep Convolutional Generative Adversarial Network (DCGAN) sampling high-quality Street View House
Numbers (SVHN), conditioned on an embedding of a desired label. The generator model is evaluated by training a fixed classifier on the generated data, and measuring the classifier performance
on an evaluation dataset. Final model is capable of generating a dataset achieving an accuracy score of close to 80%. Pictures shows generated samples in categories 0-9 after ~35 training epochs.


