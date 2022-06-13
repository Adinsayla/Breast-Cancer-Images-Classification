# Breast-Cancer-Images-Classification
The project aims to build an algorithm to automate the classification of non_IDC and IDC images.
# 1 Introduction
Breast cancer is a one of the most fatal disease in women, where Invasive Ductal Carcinoma (IDC) is most common type of breast cancer. Breast cancer is life-threatening and thousands of women died every year. The accurate identification and classification of breast cancer subcategories is an essential clinical task in medical applications, as it is a fatal disease and human lives are on stake. Manually, it requires health care experts, considerable time and experience to accurately identify and categorizing breast cancer and its subtypes. It is essential to provide computer-aided diagnosis (CAD) to healthcare staff to reduce detection error, ease their workload and improve treatment planning. The project aims to build an algorithm to automate the classification of non_IDC and IDC images. Over the last few years, there has been a tremendous popularity in usage of deep learning methods for medical applications and image analysis with a rise in success. Deep learning in the area of medical imaging can be very useful for patterns identification, classification and segmentation. In particular, we will solve binary classification problem by developing CNNs architecture and train it from scratch, and implement pre-trained CNNs based models such as VGG16 and ResNet50.
# 2 Data Exploration
The dataset that will be used is available publically on kaggle that contains Breast Cancer Histology Image samples. These images are labels as IDC positive and IDC negative. Dataset is in the form of numpy arrays which are small patches and extracted from Histology Images of breast tissues. There are many cells contain by breast tissue but only a few are IDC (+) or cancerous. Data loaded using np.load to load the numpy array in google colab.
Fig. 1 importing dataset With initial exploration of data, it was noted that the breast cancer histology dataset contains 5547 total data sample either IDC positive or IDC negative with 2759 IDC (-) Images and 2788 IDC (+) Images. We analyzed the images and the distribution of data in each class, the percentage of positive images is 50.26% hence, there is no class imbalance problem. Breast histology images dataset contain RGB images of size 50 x 50 x 3 (width, height, channels).
Fig. 1 Data description Furthermore, after importing and exploring the images visually, the data sample has shown below, the images are labeled as either 0 (non-IDC) or 1 (IDC).
Fig. 3 Data visualization
# 3 Data Preprocessing
In this section, we will prepare the data to make it suitable for building and input the model.
## 3.1 Shuffling Data
In given dataset, data samples of images labeled as 0 (non-IDC) are before the data samples labeled as 1 (IDC) as few images are shown above. To train the model effectively, avoid biased and artificial data patterns. The mixed data from both positive IDC and negative IDC class is shown in the image and compared with the image before random shuffling as follows.
Before random shuffling
After shuffling the data
Fig. 1 Data shuffling
Furthermore, various other data transformations are implemented to make it suitable as an input to the deep learning models such as data normalization to scale the data from 0 to 1 and train test split to train and validate the model.
3.2 Train Test Splitting Entire dataset split in to train and test using train_test_split from sikt_learn library. We will be using the 80% of data for training and rest 20% for testing. The image below demonstrate the implementation of splitting data.
Fig. 5 Train test split
3.3 Data Scaling In this step, we will scale/normalize train and test data from 0 to 1. The pixel values of IDC images ranges between [0, 255], therefore a traditional deep learning models works in an effective manner when the input data values ranges between [0, 1] or [-1, 1]. Data scaling is performed to transform the pixel values of IDC images in the range of [0, 1]. Moreover, one hot encoding is also implemented for class attribute.
## Fig. 6 Normalization
# 4 Building the models
To classify, either IDC positive or IDC negative, we implemented 3 Convolutional neural network (CNN) based models. 2 pre_trained models (VGG16 and ResNet50) are used to reduce the computation cost and training time. Moreover, we introduced CNN based architecture a model based on Convolutional neural network (CNN) which has proven as one of the best algorithm for image processing. Traditional CNN model consist convolution, max_pooling and fully connected layers for classification. Convolution layer usually identify various features from the image and fully connected layer which used to provide output and predict the label of image.
## 4.1 Pretrained Models
In deep learning, particularly for image processing Convolutional Neural Networks (CNN) provide promising results and most commonly used of deep learning model for medical image analysis and diagnosis (Albashish, et al., 2021). But due to training time and computation cost transfer learning is becoming more popular using pretrinaed models. To classify breast cancer, we utilized VGG16 and ResNet50 and achieved good performance as compared to proposed CNN.
## 4.1.1 VGG16
VGG16 is pretrained model, used to classify cancer breast images and outperformed all other models. The summary of VGG16 architecture is given below in fig. 7:
Fig. 7 Summary of VGG16 architecture
## 4.1.2 ResNet50
ResNet50 is another pretrained model, accuracy is less as compared to other 2 models. I used up to 500 epochs to train using breast cancer dataset without data augmentation and with
augmented data as well but the performance was same. With data augmentation all the models performed worst. Fig. 8 showing the summary of ResNet50 architecture.
Fig. 8 Summary of ResNet50 architecture
## 4.2 Fine Tuning of CNN
### 4.2.1 Convolutional Neural Network (CNN) Architecture
CNN contains of 4 convolution layers, the first layer is a 2d convolutional layer where number of filters are 32 with size of 3 x 3. The input shape of 50x50x3 is also specified at first layer in the first layer. Rectified linear unit (ReLU) is the most commonly activation function for hidden layers that is used as an activation function for all the rest of the layers except the fully connected layer
Second layer is a Max pooling layer 2x2 window. To avoid the overfitting problem, dropout layers is used with dropout rate of 0.25.
Next layer is 2d convolutional layer of 32 (3x3) filters, followed by max pooling layer of 2x2 filter and dropout layer.
The next two layers are followed same pattern of 2d cov layer, maxpooling and dropout layer.
Flatten layer is used to map the 3D feature space to 1D feature vectors and fed these 1D feature vectors to fully connected layer. Next dropout is with 0.5 dropout rate to drop 50% of the neurons randomly to prevent the overfitting. Hinton (2012) has proven the 0.5 dropout rate very effective.
Final output layer is also a dense layer consist of 2 (number of classes) neurons with sigmoid as an activation function for binary classification.
Following CNN based neural network architecture is introduced for IDC positive and IDC negative classification problem:
Input layer: [50, 50, 3]
Conv1 -> ReLu -> MaxPool: [., 24, 24, 16]
Conv2 -> ReLu -> MaxPool: [., 11, 11, 32]
Conv3 -> ReLu -> MaxPool: [., 11, 11, 32]
Conv4 -> ReLu -> MaxPool: [., 1, 1, 64]
Flatten [., 64] -> Fully connected (FC) -> [., 64]
Output layer: FC -> [., 2]
The summary of CNN architecture is given below:
Fig. 9 Summary of CNN architecture
4.2.2 CNN Training
The number of epochs are set to be 100 with batch size of 64 to a high number and uses Early Stopping method to reduce overfitting which stop the process of training when no more improvement in the performance of the model for specific number of parameters
4.3 Data Augmentation
Usually, deep learning work better with large datasets. Data augmentation is a method used to increase the number of images samples by rotating, flipping (vertically and horizontally), increasing brightness and zooming. I used data augmentation method by using
ImageDataGenerator that provides many techniques to augment data and I used following of them.
Fig. 10 Image data generator for data augmentation
Data augmentation in this way does not provided me an effective results, I used this techniques with all of the models but instead of more accurate results I observe decrease in classification accuracy.
# 5 Models Evaluation
## 5.1 Performance Metrics
The ordinary performance metrics are accuracy, precision, recall and F1 measure which is calculated to evaluate the models.
5.1.2 Confusion Matrix
All these metrics can be calculated by using the confusion metrics which consist of following values:
Positives (TP), true negatives (TN), false positive (FP) and false negative (FN)
Each row of confusion matrix shows the predicted class values while each column contains the instances in an actual class. These four values are very useful to identify misclassification and to calculate the ordinary performance metrics.
## 5.1.3 Accuracy
The model accuracy can be calculated by using sum of TP and TN and then divide it by the sum of all values in confusion matrix. Overall accuracy of models were calculated using the formula given below.
## 5.1.4 Precision and Recall
Percision and Recall can be calculated by values from confusion matrix and by using the following formula:
## 5.1.5 F1-Score
F1-Score shows the weighted average of Precision and Recall. Higher F1-Score shows the better performance of the model. For all these metrics, value near 1 shows better while towards 0 shows the worst performance of the models.
# 6 Results and Discussion
We have introduced and trained CNN model and 2 pretrianed models (ResNet50 and VGG16). All the models were trained using same following parameters given in table 2. Experiment performed using google colab and GPU to speed up the training process. The results are reported in Table. 2. Highest performance has shown by VGG (pretrained model) where accuracy is 79% which is highest as compared to CNN and ResNet50. Results of best performed algorithm is also given in Fig. 11, table 2 and table 3 which shows the comparison using different evaluation parameters and results visualization.
## Training parameters
Algorithms
Epochs
Batch_size
Optimizer
Learning rate
CNN
100
64
Adam
0.001
VGG16
100
64
Adam
0.001
ResNet50
200
64
RMSprop
1e-4
Table. 1 Training parameters
Fig. 11 Highest accuracy by VGG16
# Models evaluation
Algorithms
Accuracy
Precision
Recall
F1 score
CNN
77%
0.83 and 0.73
0.66 and 0.87
0.74 and 0.79
VGG16
79%
0.79 and 0.79
0.78 and 0.80
0.79 and 0.80
ResNet50
72%
0.69 and 0.75
0.76 and 0.68
0.73 and 0.71
Table. 2 Models evaluation
The visualizations of training and validation accuracy as well as loss has been shown in Table 3. For all 3 models.
Models
Training and validation accuracy
Training and validation loss
VGG16
CNN
ResNet50
Table. 3 Models evaluation (visualization)
# 7 Findings and Future directions
This study aims to classify breast cancer histology images either contains IDC or not. The dataset explored using image processing techniques to analyses the dataset and performed necessary preprocessing steps to prepare the data for building the models. Initially, by visualizing the data it is noted that the images are not in high resolution which also impact the model performance. This is the
reason not to use image generation algorithms for data augmentation. Three classification techniques were used to classify the images including pretrianed models. VGG16 outperformed rest of the 2 models. The performance of VGG16 was promising with low resolution data in terms of accuracy and computation cost. Overall, better classification accuracy achieved with low computation cost and time efficiency using pretrained model. In the future, semantic segmentation technique using U-Net can be used to extract the features and may provide good performance. The images quality can be improved by implementing image processing methods. In addition, deep CNN can perform better if implement with good quality images. Moreover, other pretrained models fine tuning can be very effective to increase the classification accuracy
