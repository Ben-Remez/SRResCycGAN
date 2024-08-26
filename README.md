# SRResCycGAN

A pytorch implementation of the SRResCycGAN network according to the paper: [Deep Cyclic Generative Adversarial Residual Convolutional Networks for Real Image Super-Resolution]([url](https://arxiv.org/abs/2009.03693))

## General Information
The Article Name: Deep Cyclic Generative Adversarial Residual Convolutional Networks for Real Image Super-Resolution

As a part of our academic research for our MSc degree, we researched and developed a model to perform Super Resolution tasks. 
Our project was focused on using Generative Adversarial Network (GAN) to perform this task. 
The project had two parts, a "dry" part where the primary research and planning was made, this part was more theoretical. The second part was the "wet" part where the development process and the experiments were done. 
You can find the powerpoint presentations that conclude both parts in the [Presentations](Presentations) folder in this repository. 
(If the presentations do not load up after downloading them separately, please download the whole repository and they should load up).

In our experiment process we ran training of the article's code taken from the official Git (link below) and compared the results with the results of our trained model. 
Article code Git: https://github.com/RaoUmer/SRResCycGAN?tab=readme-ov-file

The development of the model was done using google colab and the [notebooks](Notebooks) can be found in this repository. 
Please note that the model is big and requires many resources to run the training as it is configured in the notebooks. We used colab pro+ in order to run the training of the model on the A100 GPU.

We also added an src folder with adaptations to the code for anyone who wishes to use it in other environments. 

## Model Training
The license plates dataset we used was found on Kaggle and cleaned in the process of the experiments. The model had the best performance for super resolution when the training was done on 750 images from the dataset with a resolution of 2K and above.
We also trained the model on the DIV2K dataset using the weights of the best model we managed to receive from the training process on the high resolution license plates dataset.
The training on the DIV2K dataset was done to see if training on more generic images will improve the model's performance when performing Super Resolution tasks on license plates images.

## Datasets
License Plates Dataset: https://www.kaggle.com/datasets/tolgadincer/us-license-plates 
DIV2K Dataset: https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images

## Model General Architecture

## Quantitative Results

