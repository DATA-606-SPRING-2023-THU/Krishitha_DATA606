# DATA_606 - Capstone Project
- **_Professor_** : **Dr.Chaojie Wang**
- Author : Krishitha Akula
- Semester : Spring 2023
- Campus ID : VP50694

# Project Title : Brain Tumor Detection Using Deep Learning

# Project Overview

Brain tumors are one of the most common types of cancer and they can have serious implications for patient health and wellbeing. The traditional diagnostic methods for brain tumors, such as MRI, have limitations and may not always be accurate in identifying the presence of a tumor or its location. This can result in delays in treatment and a lower likelihood of successful outcomes for patients.

# Problem Statement

The problem statement for this project is to evaluate and compare the effectiveness of various neural network models in predicting the presence of brain tumors using MRI images. The primary goal of this project is to explore the potential of deep learning techniques in improving the accuracy of brain tumor diagnosis and to identify the most efficient neural network model for this purpose.

# Technologies Used
- programming language : Python.
- libraries : Pandas, Seaborn, Sklearn, Torchvision, Skimage, Matplotlib
- Techniques: Fully Connected Networks, Convolutional Neural Network(with Layer and Batch Normalization), Pretrained Model(VGG16) 

# Dataset
The data source used for this project is a dataset that is publicly available on Kaggle, a popular platform for data science competitions and projects. The dataset, which can be accessed at the URL https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection, includes a collection of MRI images of the brain that have been labeled as either containing a brain tumor or not containing a brain tumor.

The dataset consists of a total of 253 images, with 155 images containing brain tumor and 98 images without brain tumor. The images are in JPEG format and have a resolution of 256 x 256 pixels.

This dataset has been curated specifically for brain tumor detection and can be used to train and evaluate machine learning models for this purpose. It provides a valuable resource for researchers and practitioners who are interested in exploring the use of machine learning for brain tumor diagnosis. However, it is important to note that the dataset has its own limitations and biases, such as its relatively small size and potential selection bias.

Sample Image

![sample](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/f6bb198a-2383-43d1-b3df-8e0ed9cabb36)


# Data Preprocessing

* Before feeding the data to the neural network, some preprocessing steps are necessary. 
For example, the images needs to be resized to a standard size (e.g., 224 x 224 pixels), normalized to a specific range of pixel values
* Change the channel position so that each image has the same dimensions
* Since the dataset had a size limitation, I used a technique named Data Augmentation.
	* Data augmentation is a technique used in machine learning and computer vision to artificially increase the size of a dataset by applying various transformations to the existing data. It is particularly useful when working with limited amounts of data, as it helps to mitigate overfitting and improve the generalization ability of a machine learning model.

	* Data augmentation involves applying a range of operations or transformations to the original data, creating new samples that are similar but slightly different from the original ones. Some common data augmentation techniques include:

		* Flipping: Mirroring the data horizontally or vertically.

		* Rotation: Rotating the data by a certain angle.

		* Scaling: Zooming in or out of the data.

		* Translation: Shifting the data horizontally or vertically.

		* Shearing: Tilting the data along a certain axis.

		* Adding noise: Introducing random noise to the data.

		* Cropping: Selecting a smaller region from the original data.
	* In this project I have used only Rotation, Affine Transformation and Warp
	* Using this Data Augmentation, I have increased the data set size 20 fold
	
	![transform](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/f4a70c30-61c3-4795-aa43-798fcb5c6750)
	![rotate](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/98e89fdd-c9ae-4038-8c9e-74aaf9d1bcab)
	![warp](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/541a118e-58e4-4f7d-b298-f11545a96230)
* Finally split into training and testing sets.

# Modeling

## Fully Connected Network

The neural network has three fully connected layers (nn.Linear) with LeakyReLU activation functions (nn.LeakyReLU) in between. The first layer (nn.Linear) has 150528 input features, which is the flattened output of an image with dimensions 224x224 pixels. The number of neurons in the hidden layers is defined by the variable hidden_nodes. The last layer (nn.Linear) has two output neurons, which corresponds to a binary classification problem where the model predicts either class 0 or class 1.

I used leaky relu here because LeakyReLU ALLOWS a small positive gradient when the input is negative. This means that even when the input is negative, the output will still be nonzero and the gradient will still flow backward through the network, allowing for better learning and avoiding the "dying ReLU" problem.


```
first_linear= nn.Sequential(
    nn.Flatten(),
    nn.Linear(150528,hidden_nodes),
    nn.LeakyReLU(),
    nn.Linear(hidden_nodes,hidden_nodes),
    nn.LeakyReLU(),
    nn.Linear(hidden_nodes,2)
)
```

The model performed pretty well with both training and validation accuracies surging past 95 % after 10 epochs.
We also visualized the validation loss to understand how the model learnt from each epoch.

![FullyConnectedVal](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/60736db7-f18e-4772-b88d-8f4fba4b25f1)
![FullyConnected](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/8d742809-96e8-4122-a20a-e4484d3eb1c3)

## Basic CNN
Now I have created a basic CNN . A CNN, or Convolutional Neural Network, is a specialized type of artificial neural network that is designed for processing structured grid-like data, such as images or time series data. CNNs are particularly effective in computer vision tasks, including image classification, object detection, and image segmentation.

```
first_cnn= nn.Sequential(
    nn.Conv2d(3,D,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(32,1,(3),padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(224*224,2),
)
```
We can see how the network performed. It can be seen here that it did not perform good compared to the previous fully connected network. So as an attempt to increase the accuracy, I have used Batch Normalization technique.

![CNN](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/78fc9e52-078a-4caf-8401-66b3c630326f)
![CNNVal](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/4d2faf2b-6643-482b-97b0-411a4179de82)

## CNN with Batch Normalization

I have increased the number of layers and added Batch Normalization layers in the network. I have used batch normalization here thinking that since these were images and the inputs can vary and cause covariance shift, The use of batch normalization helps to overcome the issue of covariate shift, which occurs when the distribution of the inputs changes during training. This can lead to slow convergence and poor performance of the model. By normalizing the inputs, batch normalization reduces the impact of covariate shift and stabilizes the learning process.

```
second_cnn= nn.Sequential(
    nn.Conv2d(3,D,3,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(D),
    nn.Conv2d(32,32,(3),padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32,32,(3),padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32,32,(3),padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    Flatten(),
    nn.Linear(1605632,2),
)
```

But it did not make much difference so I tried Layer normalization. 

![batch](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/3ac84a73-e967-43bd-888c-ec8dbc2ea213)
![BatchVal](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/304f748c-7f46-4e54-94a1-fb70d6635736)

## CNN with Layer Normalization
Layer normalization is a technique used in neural networks to normalize the inputs to a layer across the features dimension. It is similar to batch normalization but instead of normalizing the inputs across the batch dimension, it normalizes across the feature dimension.

```
out_shape = (3, 224, 224)
width_height = (out_shape[-2], out_shape[-1])
third_cnn= nn.Sequential(
    View(-1, 3,out_shape[-2], out_shape[-1]),
    nn.Conv2d(3,D,3,padding=1),
    nn.ReLU(),
    nn.LayerNorm(width_height),
    nn.Conv2d(32,32,(3),padding=1),
    nn.ReLU(),
    nn.LayerNorm(width_height),
    nn.Conv2d(32,32,(3),padding=1),
    nn.ReLU(),
    nn.LayerNorm(width_height),
    nn.Conv2d(32,32,(3),padding=1),
    nn.ReLU(),
    nn.LayerNorm(width_height),
    nn.Flatten(),
    nn.Linear(1605632,2),
)
```

This was the best performing neural network so far. So now lets compare this with a pretrained model called Vgg16

![layer](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/c843b107-de23-48ca-a4d5-4a6329e12d88)
![layer val](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/b6b7375f-2a5e-403b-b373-b34938212f88)

## Pretrained Model - VGG16
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

```
    model = models.vgg16(pretrained=True)
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, 2)) #performing surgery by changing the output classes to 2
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
```
The pre-trained model achieved a 100% validation accuracy after just 1 epoch.

![Vgg16](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/32a399a9-0d6a-4346-ace6-d3853a9ad4aa)
![Vgg16Val](https://github.com/krishitha12/Krishitha_DATA606/assets/89949881/28679dd5-405d-472f-9c47-ee886de53f73)

# Conclusion
Although the models we designed performed well, the pre-trained model out-performed them. The constant and surging accuracy can possibly be attributed to the size of our dataset. This was a good learning experience that enlightened us to newer and efficicent techniques. An extension of this project could be using un-labelled data and trying to learn features to detect tumor.

# Repository Structure
- docs : directory for project draft and final documentations.
- src : directory for project code base.

### Links:
#### Presentation links:
- [Youtube](https://youtu.be/-xeLz4f4Z1M )
- [Powerpoint presentation](https://github.com/krishitha12/Krishitha_DATA606/blob/main/docs/Capstone_project.pptx)
