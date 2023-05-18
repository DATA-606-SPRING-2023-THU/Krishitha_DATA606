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

![]()

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

# Repository Structure
- docs : directory for project draft and final documentations.
- src : directory for project code base.

### Links:
#### Presentation links:
- [Youtube](https://youtu.be/-xeLz4f4Z1M )
- [Powerpoint presentation](https://github.com/krishitha12/Krishitha_DATA606/blob/main/docs/Capstone_project.pptx)
