# Capstone Project Data 606

## Project Title: Brain Tumor Dectection from MRI images 

## Project Overview:

Brain tumor is the growth of abnormal cells in brain some of which may leads to cancer. The usual method to detect brain tumor is Magnetic Resonance Imaging(MRI) scans. From the MRI images information about the abnormal tissue growth in the brain is identified.Many different types of brain tumors exist. Some brain tumors are noncancerous (benign), and some brain tumors are cancerous (malignant). Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors).
How quickly a brain tumor grows can vary greatly. The growth rate as well as the location of a brain tumor determines how it will affect the function of your nervous system.Brain tumor treatment options depend on the type of brain tumor you have, as well as its size and location.

## Issue of interest:

* The issue of interest for this project is to analyze and compare the performance of neural network model for predicting Brain tumor providing MRI images 

## Importance of the issue:
The issue of brain tumor detection from MRI images is of significant importance for several reasons:
* Early Detection: Early detection of brain tumors is critical for effective treatment and improved patient outcomes.
* Treatment Planning: Accurate detection and segmentation of brain tumors from MRI images is important for treatment planning.
* Patient Management: MRI images can provide valuable information for monitoring the progression of brain tumors over time.

## Objective:

Specifically, the objectives of brain tumor detection from MRI images include - Accurate identification of the location and size of the tumor, which can help in treatment planning and surgical interventions.Development of computer-aided diagnostic tools to assist radiologists in the interpretation of MRI images and improve the accuracy and efficiency of brain tumor detection.

## Data Source:

The data used for this project will be a dataset from the Kaggle platform https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection. The dataset contains 155 images of tumor images and 98 images of tumor-less images.

## Techniques/Models

The techniques/models that will be used in this project are Data Augmentation to make the most of our few images. 
* I will augment them via a number of random transformations such that the model would never see the exact image twice. 
* I am planning to construct a novel neural network model, CNN, and later compared it with a pre-trained model(VGG16) in terms of accuracy, validation loss, precision, and other metrics.

## Outcomes:

This project can help detect tumors at early stages which can increase the chances of a patient’s recovery after treatment.
In the last decade, we have noticed a substantial development in medical imaging technologies, and they are now becoming an integral part of the diagnosis and treatment processes.
