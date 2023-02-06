A brain tumor is a mass or growth of abnormal cells in your brain. Many different types of brain tumors exist. Some brain tumors are noncancerous (benign), and some brain tumors are cancerous (malignant). Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors).

How quickly a brain tumor grows can vary greatly. The growth rate as well as the location of a brain tumor determines how it will affect the function of your nervous system.

Brain tumor treatment options depend on the type of brain tumor you have, as well as its size and location.

The aim of this project is to classify MRI images of the brain for tumor and tumorless images. The data source for this data is https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection. The dataset contains 155 images of tumor images and 98 images of tumor-less images. I plan to use Data Augmentation to make the most of our few images. I will augment them via a number of random transformations such that the model would never see the exact image twice. I am planning to construct a novel neural network model, CNN, and later compared it with a pre-trained model(VGG16) in terms of accuracy, validation loss, precision, and other metrics.

This project can help detect tumors at early stages which can increase the chances of a patient’s recovery after treatment. In the last decade, we have noticed a substantial development in medical imaging technologies, and they are now becoming an integral part of the diagnosis and treatment processes.
