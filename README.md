# AutoEncoder
Autoencoders are neural networks. Neural networks are composed of multiple layers, and the defining aspect of an autoencoder is that the input layers contain exactly as much information as the output layer. The reason that the input layer and output layer has the exact same number of units is that an autoencoder aims to replicate the input data. It outputs a copy of the data after analyzing it and reconstructing it in an unsupervised fashion.

The data that moves through an autoencoder isn’t just mapped straight from input to output, meaning that the network doesn’t just copy the input data. There are three components to an autoencoder: an encoding (input) portion that compresses the data, a component that handles the compressed data (or bottleneck), and a decoder (output) portion. When data is fed into an autoencoder, it is encoded and then compressed down to a smaller size. The network is then trained on the encoded/compressed data and it outputs a recreation of that data.

So why would you want to train a network to just reconstruct the data that is given to it? The reason is that the network learns the “essence”, or most important features of the input data. After you have trained the network, a model can be created that can synthesize similar data, with the addition or subtraction of certain target features. For instance, you could train an autoencoder on grainy images and then use the trained model to remove the grain/noise from the image.

**AutoEncoder Architectur**

![image](https://user-images.githubusercontent.com/86812576/172150842-06115dca-4e5e-4d62-9937-7cd7f30d6ca7.png)

# Import Package
import common packages:
- import numpy as np
- import matplotlib.pyplot as plt
- import torch
- from torch import nn, optim
- from jcopdl import Callback, set_config
- from torchvision import datasets, transforms
- from torch.utils import DataLoader
- 

# Import Dataset
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset. The original creators of the database keep a list of some of the methods tested on it. In their original paper, they use a support-vector machine to get an error rate of 0.8%.

Extended MNIST (EMNIST) is a newer dataset developed and released by NIST to be the (final) successor to MNIST. MNIST included images only of handwritten digits. EMNIST includes all the images from NIST Special Database 19, which is a large database of handwritten uppercase and lower case letters as well as digits. The images in EMNIST were converted into the same 28x28 pixel format, by the same process, as were the MNIST images. Accordingly, tools which work with the older, smaller, MNIST dataset will likely work unmodified with EMNIST.

Sample images from MNIST test dataset

![image](https://user-images.githubusercontent.com/86812576/172053706-ea5f4c0d-ed17-4e3e-8d80-9fa93bcd0b3b.png)

This data has been structured in tabular form. consists of 2000 lines and 785 and 9 labels in it. 785 columns are the pixels of each label. each label has 28x28 pixels.

# Dataset & Dataloader

![Screenshot 2022-06-05 205733](https://user-images.githubusercontent.com/86812576/172054258-0e75aed9-bd29-46dd-9004-e7c60d31ddbc.png)

In this step I use a batch size of 64, while for the transform I only use grayscale for the train and test data.

# Architecture

![Screenshot 2022-06-05 211421](https://user-images.githubusercontent.com/86812576/172054890-274de534-4ca2-47e8-8e3b-d041d14d7f32.png)

For the architecture I just use a simple neural network.

![Screenshot 2022-06-05 210346](https://user-images.githubusercontent.com/86812576/172054446-31349c20-09d0-4a43-83ed-92a2db304074.png)

The encoding starts from 784 neurons to 512 to 256 to the latent space (z_size). And the activation used is sigmoid (0 to 1). The latent space if possible is limited.

As for decoding, it is the opposite starting from the latent space to 256 to 512 and 784 also with sigmoid.

# Config
I did tuning in the config for z_size as much as 32 which means latent space. And the batch size is 64.

# Training Preparation -> MCOC

![Screenshot 2022-06-05 212229](https://user-images.githubusercontent.com/86812576/172055234-177fe073-046e-4e25-a0d2-696ad0badbe3.png)

The model uses the AutoEncoder which was created with the appropriate z_size in the config. For criterion, because the activation is sigmoid, then I use BCE Loss, with the optimizer commonly used, namely AdamW (Adaptive Momentum with Weight Decay).

# Training and Result

![Screenshot 2022-06-06 174617](https://user-images.githubusercontent.com/86812576/172146623-ea4924ad-c546-4666-9fd8-9100ebeee062.png)

The result looks a small loss, but looks a little overfit, that's natural because it reduces an image that has never been seen.

# Sanity Check

![image](https://user-images.githubusercontent.com/86812576/172056249-6fc38070-9ff6-437f-bda4-24a0585b2e0f.png)

In this step, we will immediately see the prediction of the image generated by the model.
In the image shown, it can be seen that the top row is a sample image of the data. After encoded, latent space is generated, which is a random code (second line), we may not understand the meaning of these codes, but the machine does. Then do the decoding (third line) that is, the original image is reversed to the original image. It turns out that the pictures are very similar although there are slight differences.

# Denoising Auto Encoder
![Screenshot 2022-06-06 154541](https://user-images.githubusercontent.com/86812576/172128095-8371f14a-a130-43cd-8d6e-3d4acce012e0.png) 

The idea is simple, which is to add noise to the input image, then when encoded it only sacrifices the noise, so that when decoded, the image returns to its original state without any lost information.

![image](https://user-images.githubusercontent.com/86812576/172146741-d7c04744-2adf-4f8e-8111-dc1bc6c96e12.png) 
![image](https://user-images.githubusercontent.com/86812576/172146834-44fd9d47-e0a5-487d-8a8c-cefaf2e88201.png)

The image on the left is the image with added noise when encoded. The image on the right is the output of the input with added noise.

# Sanity Check with Denoising

![image](https://user-images.githubusercontent.com/86812576/172148221-88205664-a5a7-4722-b1aa-85fd5e0be1a6.png)

At first (first row) the original data is added noisy, then it is encoded by removing unnecessary information. In this case, what is meant by unimportant information is its noisy. So that the results can be seen (third row) is successful. The image containing the noise is cleaned, because it is as if the machine is not only tasked with reconstructing it, but is removing the noise.
