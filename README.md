# APPLICATIONS-OF-LLM
In this project, we will be going through some of the core algorithms used in the architecture of LLM and some necessary libraries used in it. 
# TASK 1: DESIGNING CNN ARCHITECTURE
We will be using the fashion_mnist dataset which consists of nearly 70,000 images.
First, We will be dividing the data into train and test datasets using the load_data() function. Normalizing all the pixel values in between 0 and 1 would make easier to handle the pixels and applying operations on them.
Neural networks is a sequential model which consistes of different layers i.e. Convolutional Layer, Pooling Layer, Dense Layer.
Flattening the image matrix is necessary to get an input for the dense layer, in this layer learning of the parameters is done.
Then compiling the model where the weight adjustments and evaluation of metrics like accuracy, loss are done.
Now, fitting the model onto the learned parameters which outputs the metrics required.Two types of accuracies are noted namely, train_accuracy and test_accuracy.
 In the process of fitting the model two things other than the normal fitting can happen i.e. UNDERFITTING and OVERFITTING.
Underfitting is when train_accuracy or test_accuracy or both are very low.
Overfitting is when train_accuracy and test_accuracy show a significant gap in between them.
 Predicting the test images and use them to obtain a classification report which provide information about the metrics like Precision, F1_Score, Recall.
IMPROVING THE ACCURACY
Using 3 Convolutions and 2 Poolings accuracy obtained was 0.883. To increase this accuracy first try was made by changing the Convolutions and Poolings to 2 and 1, then the accuracy changed to 0.912.
  Decreasing the number of convolutional layers from 64 to 32 resulted in the accuracy of 0.9157.
Keeping the convolutional layer constant and changing the dense layers to 32 resulted in the accuracy of 0.9166.
But in all this Overfitting was noticed as the train_accuracy and the test_accuracy was seen to have a significant gap.
To address this issue a dropout layer was used but it was of no use. Now using the L2_regularization technique the problem of overfitting was solved upto some extent but finally the accuracy which was of the value 0.9166 reduced to 0.8975. 
To make the model give more accuracy and lower the problem of overfitting we use the technique called TUNING, this process demands for a range of input parameters like units, dropout, learning rate and accuracy.
Then it searches for the best possible parameters in the provided values and gives the best accuracy.
This method increased the accuracy to 0.953 or around that value and the overfitting was also no noticed.
# TASK-2 
In the second task we learnt how to apply LSTM models on time-series data and explore how changes in model architecture and hyperparameters affect the results. We started with the provided sample code, which used LSTMs to predict sequential data, and then went on to experiment with the model architecture by adding more LSTM layers and dense layers, adjusting the number of neurons, and incorporating Dropout layers to study their impact on the model's complexity and performance.
From this task we learnt about RNNs and LSTMs. Their architecture and working, how they are very useful in processing sequential data and how  we can apply it on real world data.
# TASK-3
In this task, we will be going through the encoder and decoder models using attention mechanisms. Also, we will go through the tuning the hyperparameters and layers.The Encoder-Decoder architecture is a framework for handling sequence-to-sequence tasks, such as translation and summarization. It maps variable-length input sequences to variable-length outputs using two main components.
Basic applications of the transormer model are 
Machine Translation: E.g., Google Translate.
Text Summarization: Generating concise representations of documents.
Speech-to-Text Conversion: Used in virtual assistants like Siri.

Tuning Hyperparameters: This refers to adjusting settings like learning rate, batch size, and optimizer to improve model performance during training. These parameters are set before training and impact how the model learns.

Adjusting Layers: This involves changing the architecture of the neural network by adding, removing, or modifying layers (e.g., dense, convolutional, or recurrent layers) to better capture patterns in the data.

