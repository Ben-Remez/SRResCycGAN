The Article Name: Deep Cyclic Generative Adversarial Residual Convolutional Networks for Real Image Super-Resolution

In our experiment process we ran training of the article's code taken from the official Git (link below) and compared the results to our trained model. 
Article code Git: https://github.com/RaoUmer/SRResCycGAN?tab=readme-ov-file

The license plates dataset we used was found on Kaggle and cleaned in the process of the experiments. The model had the best performance for super resolution when the training was done on 750 images from the dataset with a resolution of 2K and above.
We also trained the model on the DIV2K dataset using the weights of the best model we managed to receive from the training process on the high resolution license plates dataset.
The training on the DIV2K dataset was done to see if training on more generic images will improve the model's performance when performing Super Resolution tasks on license plates images.

License Plates Dataset: https://www.kaggle.com/datasets/tolgadincer/us-license-plates 
DIV2K Dataset: https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images


