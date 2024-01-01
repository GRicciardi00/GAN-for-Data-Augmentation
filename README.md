# GANs for Data Augmentation in Image Classification

This project explores the application of Generative Adversarial Networks (GANs) for data augmentation in the context of image classification. GANs play a key role in improving machine learning models by creating synthetic data, preventing overfitting, and enhancing the model's ability to perform well on new, unseen data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)

## Project Overview

Data augmentation is crucial for improving machine learning models by diversifying training datasets and preventing overfitting. This project focuses on using GANs, specifically DCGAN, WGAN, and VAE-GAN, to generate synthetic data for enriching an image classification dataset.

### Objectives
  1.  **Data Augmentation with GANs:** 
	   - Implement and compare three distinct GAN architectures for generating synthetic images: DCGAN, WGAN, and VAE-GAN.
	   -  Explore the synergies between GANs and data augmentation techniques to enrich and balance the training dataset.
  2.   **Image Classification:**  
		  - Utilize a Convolutional Neural Network (CNN) for image classification.- Evaluate the impact of GAN-generated images on the CNN's classification performance. 
 ![Representation of application architecture](https://github.com/GRicciardi00/GAN-for-Data-Augmentation/blob/main/project_architecture.png)
## Dataset

The provided dataset comprises 164 color images, each sized 256x256 pixels, depicting various angles of 3D models. The images represent 3D models of cows and horses, with diverse angles to capture different perspectives.Some pre-processing operations have been carried out (resize, data augmentation, etc.).
![Images generated with trained WGAN model](https://github.com/GRicciardi00/GAN-for-Data-Augmentation/blob/main/Generated_images.png)

## Project Structure
 The project structure is designed for clarity and ease of use:
 - `main.py`, orchestrates the training of GANs, data augmentation, and the subsequent image classification process.
 - `CNN.py` module is dedicated to the Convolutional Neural Network (CNN) used for image classification. It defines the architecture of the CNN and includes the necessary training and evaluation procedures.
 - `GAN_models.py` module encapsulates the implementation of three GAN architectures: DCGAN, WGAN, and VAE-GAN.
## Installation
To set up the project locally, follow these steps:
```bash
# Clone the repository or download it

# Install dependencies

 - Pytorch (torch, torch vision)
 - Sklearn
 - Matplotlib
 - Seaborn

# Run the main script
python main.py

#Follow The instruction on the terminal 
