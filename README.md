# **SIEMENS HEALTHINEERS** 
**CAPSTONE PROJECT - University of British Columbia**
---------------------------------------------------------------

# COMPUTER VISION ALGORITHM FOR QUALITY CONTROL OF SENSOR MANUFACTURING PROCESS


This package was developed to implement quality control of Siemens Medical Sensor during their sensor manufacturing process.
The medical sensors in our project are sometimes prone to defects while manufacturing which can be viewed by the presence of bubbles or voids in their images. 
These defected sensors have to be removed as they can further lead to medical error/ critical medical error when used for decsion making processes by doctors or other health care workers. To solve this issue we have developed a package which uses computer vision classification algorithm like CNNs to detect and classify defected sensors from the normal ones. 

### **Sensor Images and Classes:**

| S.No. | Image Classes | Images | Description |
|-------|---------------|---------|-------------|
|1.| Void | ![image](/images/voids.jpg) | Sometimes due to misalignment in the sensor position the layers printed over one another can be prone to voids. These voids depending on their size may lead to some serious medical errors and hence should be rejected as defects |
|2. | Anomaly | ![image](/images/anomaly.jpg) | During the manufacturing process, the sensors can be prone to bubbles, debris from enviornment or even fabric from the workers. These sensors have to be rejected |
|3. | Normal | ![image](/images/normal.jpg) | These sensors donot have any defects in them and hence pass our quality check |


### **Project Methadology:**

The first step of our project involves pre-processing the images to remove the background noise. Then we have to undergo a dynamic cropping process to extract our region of interest. This is acheived by creating a shallow CNN for classifying the images on the basis of their position. Then the images are pre-processed to highlight the essential features by improving its contrast. The images are then passed through a deep CNN network-ResNet to classify the images into their category. 


![image](/images/PIPE_LINE.PNG)


### **The Image classification package:**

The package performs a binary classification where it uses a pretrained model to predict/detect normal images from anomalies/voids. The package has the following folder structure. 

<img src="/images/CodeStructure.png" alt="drawing" width="300"/>

The package has separate modules for image pre-processing, dynamic cropping and the classification module. The models are trained with optimum number of images using a computer with GPU and then store in the folder ModelArchive. The pretrained models are then used for making the required predictions for the new images. 

The run_classification_pipeline.py file constitutes the entire pipeline of the process.  The file requires three different location inputs ??? input to the RAW images, the input to the saved model for dynamic cropping and the input to the saved model for classifcation model. 

The file executes the following processes: 

* Get the parent directory location of the RAW images 

* Run a dynamic cropping system to extract and return the region of interest (ROI)

* Change the contrast of the images depending on which line the raw images belong to and send the pre-processed image to the model 

* Use the saved ResNet model to predict if a sensor in the image is normal/anomaly 

* Save the predictions in a csv file for future reference  











