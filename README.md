# SVHN Image and Label Synthesis 
_3rd Place at EESTec AI Marathon, ETH Zurich 2024_

We implement a conditional Deep Convolutional Generative Adversarial Network (DCGAN) sampling high-quality Street View House
Numbers (SVHN), conditioned on an embedding of a desired label. The generator model is evaluated by training a fixed classifier on the generated data, and measuring the classifier performance
on an evaluation dataset. Final model is capable of generating a dataset achieving an accuracy score of close to 80%. (PyTorch)

