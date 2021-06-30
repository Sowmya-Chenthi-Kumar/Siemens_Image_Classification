# IMAGE PRE-PROCESSING FILE:

The image pre-processing files is used for the required pre-processing and preparation of the images. The pre-processing techniques are used for removing the background noise present and for enhancing the features of the images for classification. The preparation techniques are used for preparing the dataset for training the model. 

**Data Preparation:**

The raw images are simple camera images of large size. Hence to make training of the networks easy, we have to resize the images, crop them, augment and contrast them. 

The images are resized to a constant size of 224 X 224. The cropping is done using synamic cropping system which is present as a seperate module. The images are then augmented-rotated across all the four different quadrants. Finally we have the dataset prepared. 

**Preprocessing Images:**

A number of pre-processing techniques were tried to see what fits our images the best. The bubbles and voids present in the images are very small. Hence, using techniques which improve the mild features and edges in the images are required. 


After trying various techniques [sharpness, contrasted, grayscaled, edges, inverted, RGB Split], contrast varied images show the best results for our images. 

The processed images have been used for both training the network and also used when predicting.






