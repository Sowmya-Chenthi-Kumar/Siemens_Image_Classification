# CLASSIFICATION MODEL 

This module has the required code for training the classification model. After we have the images cropped, resized and preprocessed, we have to train the neural network model. 

Instead of designing the a convolutional model from scratch we have used existing networks and incorporated that into our system using transfer learning. Transfer learning is problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. 

![image](/images/Transfer_learning.jpg)

We have tried VGG networks, SqueezeNet, DenseNet, InceptionNet and ResNet. Among all other networks the ResNet gave us the best results of over 95% accuracy for image classification of our sensor images. 

The module is designed in a way in which the number of classes are flexible and can be changed depending upon the granularity of classes of image required. We can also choose the model which we have to train. Now depending upon our choice the model can be trained and the trained model can be stored under ModelArchives. 