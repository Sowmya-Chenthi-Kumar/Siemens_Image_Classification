# DYNAIMC CROPPING SYSTEM:

In our sensor images there is a large amount of external noise present in the system due to variations in the ambient lighting conditions or presence of the neighbouring sensor in the required image. Hence in-order to remove the background noise it is essential to crop the image and just remove the sensor out of this image. 

One major problem faced with respect to cropping is efficiently locating the sensor within the image.Sometimes the sensors tend to be more towards the left or the right. Hence as a work around to this problem, we have implemented a dynamic cropping system. 

The dynamic cropping developed by us involves repurposing a simple shallow CNN. The input to this CNN is a dataset which is divided based on the sensor location. Three folders - left, right, centre is created manually depending on the sensor location. Then these folders are fed as an input to the CNN hence training the model to classify the images on the basis on the sensor location. Once the repurposed CNN classified the images, we can crop them with static points depending on the location of the sensor. 

![image](/images/Dynamic_model.PNG)

This process effectively extracts the region of interest and removes the background noise effectively.