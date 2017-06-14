# **Traffic Sign Recognition** 

---

## Goals

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/violin_set.png "Visualization"
[image2]: ./images/greyscale_prepro.png "Grayscaling"
[image3]: ./images/30kmh.jpg "Traffic Sign 0"
[image4]: ./images/50kmh.png "Traffic Sign 1"
[image5]: ./images/60kmh.png "Traffic Sign 2"
[image6]: ./images/oneway.jpg "Traffic Sign 3"
[image7]: ./images/pedestrian.jpg "Traffic Sign 4"
[image8]: ./images/stop.jpg "Traffic Sign 5"

## Rubric Points
In this writeup, I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

This is my writeup for the Traffic Sign Classifier Project! And, here is a link to my [project code](https://github.com/yeseen/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### basic summary of the data set.
I used the python and numpy.shape to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Please refer to the Jupyter notebook for the code used to get these nnumbers.

#### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a violin chart showing how the data is distributed in the Validation, Test, and Training sets. The y-axis is the labels of the traffic signs and the width of the blue area of each set is the relative frequency of the sign in the set. 

![alt text][image1]

It looks like there are more examples for the signs with labels between 1 and 15, so the Model is expected to be able to recognize these signs better. These are mostly the speed limit signs. We will see if that's the case.

In this visualization, we can also see that the three sets have a very similar distribution so the validation accuracy will not be skewed for a particular sign. 

### Design and Test a Model Architecture

#### Pre-processing
I decided to convert the images to grayscale because I wanted to see how well the LeNet-5 network would do without the color information. To do so, I multiplied the red, green, and blue values of each pixel of each image by their luminance-preserving coefficients from [Wikipedia](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale):
>  X_train_grey=np.dot(X_train, [0.2126, 0.7152, 0.0722])[:,:,:,np.newaxis]

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data so that the weight of the network don't blow up by trying to do the normalization themselves. This is done simply by substracting 128 from each pixel value and diving it by 128:
> X_train_norm=(X_train_grey-128*np.ones(X_train_grey.shape[1:4]))/128

However, I found in practice that networks using color inputs performed better. So my final preprocessing consisted of only the normalization step:
> X_train_norm=(X_train-128*np.ones(X_train.shape[1:4]))/128


#### final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 24X24X24      									|
| RELU					|												|
| Fully connected		| Input = 13824. Output = 1200.       									|
| Fully connected		| Input = 1200. Output = 150.       									|
| Fully connected		| Input = 150. Output = 84.       									|
| Fully connected		| Input = 84. Output = 43.       									|

Really exploring the potential of a fully connected neural network with 4 fully connected layers.
 

#### Training the model

To train the model, I used the same optimizer as the LeNet-5 solution, and I adjusted the learning rate (rate) , number of epochs (EPOCHS), and batch size ) (BATCH_SIZE) to reach the desired accuracy. The final values used are:
> rate = 0.0005
> EPOCHS = 25
> BATCH_SIZE = 120

#### Approach

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.939 
* test set accuracy of 0.929

An iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture was the LeNet-5 one from the lab solution. It was used as a starting model.
* What were some problems with the initial architecture?
The Validation Accuracy was 0.892 below the required accuracy of 0.93.
* How was the architecture adjusted and why was it adjusted? 
I tried many adjustments such as doing the convolutions on separate color channels and removing a convolution layer, but what I worked the best is removing pooling layers and adding more fully connected layers
* Which parameters were tuned? How were they adjusted and why?
The learning rate and the number of epochs were the parameters that I adjusted the most. As we're skiing down the error mountain, I thought of the learning rate as the parameter that allows us to shred past tree holes when it's big (avoid being stuck in locally optimal accuracies) and to avoid going through big jumps that lends us in bad spots when it's small (avoid overfitting the netrwork with the current batch)
* What are some of the important design choices and why were they chosen? 
I kept the convolution layers for their importance in applying the network's "knowledge" to different parts of input images while I added fully connected layers to harness the black (box) magic of deep learning.


### Testing the Model on New Images

#### Test images

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


