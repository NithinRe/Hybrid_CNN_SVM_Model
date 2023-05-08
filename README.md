# A Hybrid CNN-SVM Model for Improved Real vs. Fake Image Classification

## Abstract
In this project, we explored two different approaches for image classification: Deep Learning using a Convolutional Neural Network (CNN) and Support Vector Machines (SVM) with features extracted from the CNN model. Our objective was to distinguish between real and fake images. We used a custom dataset for our experiments. Our results showed that the SVM model, trained on features extracted from the CNN, outperformed the standalone CNN model in terms of accuracy, precision, and recall. In this report, we present the details of our approach and results, along with the challenges faced during the project.

## Introduction
Image classification is an important task in computer vision, with applications in many fields, including security, medicine, and entertainment. The ability to distinguish between real and fake images is particularly relevant in today's digital age, where images can be easily manipulated and misused. In this project, we explored two different approaches for real vs. fake image classification: Deep Learning using a CNN model and traditional machine learning using an SVM classifier trained on features extracted from the CNN model.

## Dataset Preparation and Preprocessing
For our experiments, we used a custom dataset containing real and fake images. The dataset was loaded from different directories for training, testing, and validation purposes. We used OpenCV to read the images and converted their color space from BGR to RGB format. We then combined the real and fake images to create the final training, testing, and validation sets.

## CNN Model
We explored four different CNN models with different regularization techniques: no regularization, L1 regularization, L2 regularization, and dropout with L2 regularization. We found that the CNN model with L2 regularization performed the best on our dataset. The model contained four convolutional layers with batch normalization and max-pooling, followed by a fully connected dense layer with dropout and a final output layer with a sigmoid activation function. We used the Adam optimizer with a learning rate of 1e-4 and binary cross-entropy loss. During training, we used early stopping, model checkpoint, and learning rate reduction on plateau callbacks. The model was trained for 100 epochs with a batch size of 64.

## SVM Classifier
We trained an SVM classifier using features extracted from the dense layer of the pre-trained CNN model. We used Principal Component Analysis (PCA) and the SVM classifier from scikit-learn for this purpose. We trained an RBF kernel SVM classifier on the extracted features from the validation set.

## Evaluation and Comparison
We evaluated the performance of both models using accuracy, precision, and recall metrics on the test set. We also generated a confusion matrix and a classification report for the SVM classifier. Our results showed that the SVM classifier, trained on features extracted from the CNN model, outperformed the standalone CNN model in terms of accuracy, precision, and recall.

## Challenges
During the project, we faced several challenges, including a limited dataset, overfitting, feature extraction, and dimensionality. We overcame these challenges by using regularization techniques, early stopping, and feature extraction using PCA.

## Conclusion
In conclusion, our experiments showed that the SVM classifier trained on features extracted from the CNN model outperforms the standalone CNN model in real vs. fake image classification. Our findings highlight the potential of combining deep learning and traditional machine learning techniques for improved classification performance. We hope that our approach and results will be useful for future research in this area.
