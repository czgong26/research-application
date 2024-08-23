# Task Two

## Overview

For this assignment, I trained a variational autoencoder (VAE) to generate images of the digits 0-9. Trained on the MNIST dataset, the goal is to explore how the size of latent representation affects the ability of the VAE to generate clear and accurate images.

## Architecture

There was no required architecture for the VAE, so I used a common type of VAE architecture that consists of: three layers for the encoder (input, mean, log variance), reparameterization, two linear layers for the decoder, and forward pass.

The loss function looks at the difference between the original and reconstructed images via binary cross-entropy (BCE). KL Divergence is used to regularize the latent space.

## Latent Dimensions and Results

With respect to the task, I chose three latent dimensions to see how representation will impact the VAE’s performance: 5, 10, 50. 

To test for underfitting, a latent size of 5 provides constraints that force the VAE to only use a limited amount of complexity. 10 serves as a middle ground, to see performance when the VAE has a moderate amount of information it can retain. To address potential overfitting, we test the VAE with a latent size of 50.

Given the plots and images, there is a significant reduction in reconstruction loss between latent sizes 5 and 10. In the reconstructed visualizations at the end, the reconstructed images at latent dimension 5 are not as clear as the MNIST images, but are legible. There are mix-ups, such as between numbers like 4 and 9. For latent size 10, the images gradually become clearer and have better accuracy. There may be occasional errors. Between latent sizes 10 and 50, the difference in reconstruction loss is less severe. In terms of the reconstructed images, there is a much higher accuracy and imitation of the MNIST data – nearly identical in many instances.

For further consideration, it may not be necessary to use as high of a latent dimension like 50. A number between 10 and 50 in latent size may work as well, which could be beneficial in reducing computation cost and risks of overfitting.

## Citations

I referenced the following code and articles to help understand and develop my VAE.

https://github.com/williamcfrancis/Variational-Autoencoder-for-MNIST

https://botorch.org/tutorials/vae_mnist

https://krasserm.github.io/2018/04/07/latent-space-optimization/

https://github.com/pytorch/examples/tree/main/mnist

