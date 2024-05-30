# SVHN Image and Label Synthesis 
_3rd Place at EESTec AI Marathon, ETH Zurich 2024_

We implement a conditional Deep Convolutional Generative Adversarial Network (DCGAN) sampling high-quality Street View House
Numbers (SVHN), conditioned on an embedding of a desired label. The generator model is evaluated by training a fixed classifier on the generated data, and measuring the classifier performance
on an evaluation dataset. Final model is capable of generating a dataset achieving an accuracy score of close to 80%. 


## Architecture
![image](https://github.com/radimurban/svhn-synthesis/assets/78273894/9683f6fe-c39c-4cf6-b2b4-3d57559043fb)


We use conditional deep convolutional GANs. This means we condition the generator on a specific label. This translates to appending the initial noise by the embedding of the desired label. In the generator model, we generate the image that should be close to the actual image.
```Python
def forward(self, x, label):
  x = x.reshape([x.shape[0], -1, 1, 1])

  # Get the label embedding
  label_embed = self.label_embedding(label)
  label_embed = label_embed.reshape([label_embed.shape[0], -1, 1, 1])

  # Append the to the input
  x = torch.cat((x, label_embed), dim=1)

  # Apply the convolutional layers
  x = F.relu(self.tconv1(x))
  x = F.relu(self.tconv2(x))
  x = F.relu(self.tconv3(x))
  x = torch.tanh(self.tconv4(x))
  return x
```

On the other hand, the discriminator returns a probability that a image is real (as opposed to generated).
```Python
def forward(self, x, label):
  alpha = 0.2
  label_embed = self.label_embedding(label)
  label_embed = label_embed.reshape([label_embed.shape[0], 1, self.image_size, self.image_size])
  x = torch.cat((x, label_embed), dim=1)
  x = F.leaky_relu(self.conv1(x), alpha)
  x = F.leaky_relu(self.conv2(x), alpha)
  x = F.leaky_relu(self.conv3(x), alpha)
  x = torch.sigmoid(self.conv4(x))
  return x.squeeze()
```

In the training loop, we then train both discriminator and the generator.

```Python
# Loading data
x_real, x_label = data
z_fake = torch.randn(BATCH_SIZE, Z_DIM)

# Generate fake data
x_fake = gen(z_fake, x_label)

# Train Discriminator
fake_out = dis(x_fake.detach(), x_label)
real_out = dis(x_real.detach(), x_label)
d_loss = (loss_fn(fake_out, fake_label) + loss_fn(real_out, real_label)) / 2

d_opt.zero_grad()
d_loss.backward()
d_opt.step()

# Train Generator
fake_out = dis(x_fake, x_label)
g_loss = loss_fn(fake_out, real_label)

g_opt.zero_grad()
g_loss.backward()
g_opt.step()
```
The trained model generates a dataset close to the actual SVHN dataset as measured on the a fixed classifier. Pictures below shows generated samples in categories 0-9 after ~35 training epochs.

![image](https://github.com/radimurban/svhn-synthesis/assets/78273894/5a237304-59d2-42f4-900a-4ca3c4aaa099)
