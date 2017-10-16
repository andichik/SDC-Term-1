## Traffic Sign Recognition Project Writeup (andichik)
---

**Build a Traffic Sign Recognition Project**

This is project 2 of Udacity's Self-Driving Car Nanodegree. The objective of this project is to utilize what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. My model is trained, validated, and tested with [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). I later test the model with my own, provided images and analyze the model's performance.

[//]: # (Image References)

[image1]: ./images/dataset_visualization.png "Example Distribution"
[image2]: ./images/example_visualization.png "Example Overview"
[image3]: ./images/fake_data.png "New Image"
[image4]: ./images/fake_data_distribution.png "Updated Distribution"
[image5]: ./images/new_images.png "New Images from Web"
[image6]: ./images/new_images_predictions.png "Predictions on Web Images"

### Data Set Summary & Exploration

#### 1. Basic Dataset Summary

The image data provided by INI Benchmark consisted of:
* 34799 training examples
* 4410 validation examples
* 126300 testing examples
* 43 Different Traffic Sign types
* 32 x 32 x 3 (RGB) Images

#### 2. Visualization of the dataset.

This is a bar graph of the number of examples provided for each type of sign:

![alt text][image1]

We can tell that the dataset is bias because there are more examples of certain types of signs than other types. Ways to work around this bias dataset will be discussed.

Here is an example of each type of traffic signs I used as a reference:

![alt text][image2]

### Design and Test Model Architecture

#### 1. Preprocessing Data

Preprocessing the data before training, validating, and testing model is as important as designing the model itself. My preprocess pipeline has three main steps for the training data.

##### a. Converting RGB to Grayscale

Color plays a very little role in identifying traffic signs, so I converted all data (including validation and test dataset) to grayscale.

##### b. Generating 'Fake Data'

I compensated for the bias training data by generating more data by apply random transformation to each of the training images that under-represented. Transformations I applied include:

* Rotatation
* Translation
* Scaling
* Warping
* Brighness-adjusting

The model will treat the 'fake data' as different data as the original.

These are some example results of passing an image through a random transformation:
![alt text][image3]

Here is the distribution of the datasets after adding the additionally generated images

![alt text][image4]

##### c. Normalizing

Normalizing converts the image data from [0, 255] scale to [-1, 1] scale.

#### 2. Model

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Grayscale image                       | 
| 1: Convolution 5x5    | 1x1 stride, valid padding, outputs 28x28x6    |
| 1: RELU               |                                               |
| 1: Max pooling        | 2x2 stride, outputs 14x14x6                   |
| 1: Dropout            |                                               |
| 2: Convolution 5x5    | 1x1 stride, valid padding, outputs 10x10x16   |
| 2: RELU               |                                               |
| 2: Max pooling        | 2x2 stride, outputs 5x5x16                    |
| 2: Dropout            |                                               |
| A: Convolution 5x5    | From Layer2:  1x1 stride, outputs 1x1x400     |
| A: RELU               |                                               |
| A: Fully-connected    | outputs 400                                   |
| B: Fully-connected    | From Layer 2: outputs 400                     |
| 5: Fully-connected    | Concatenate Layer A & B                       |
| 5: Dropout            |                                               |
| 6: Softmax            | outputs 43                                    |

This model is based on Pierre Sermanet and Yann LeCun's paper on the same subject, found [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

#### 3. Training Method

To train the model, I ran 35 epochs of Batch size of 128, and a learning rate of 0.0009.

For regularization, I used dropout with a static probability of keeping of 0.7. I considered using l2 regularization as well, but discovered very minimal or negative impact on the results.

The cost function (loss) to be minized is Softmax Cross Entropy which weight the discrepency between the predicted probability for each label (logits) and the actual labels classification.
The optimizer I used is the same used for the LeNet lab, the adam optimizer which is a type of Gradient Descent Opimization Algorithm.

#### 4. Approach Taken

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I initially used the same model architecture as the LeNet. The model did surprisingly well as predicted, but I immediately noticed that the model was overfitting; the training accuracy was much greater than the validation accuracy. This discovery led me to add dropout and consider adding l2 regularization to avoid overfitting. I had to increase number of epochs to compesate for the dropout. This is also where I decided to generate more image data to overcome bias datasets, which allows the model to be trained neutrally across all types of traffic signs.

While reading Sermanet and LeCun's preprocessing methods, I decided to migrate over to their model architecture because this model was designed specifically for this Traffic Sign Classification problem and because LeCun's developed the model I was using already. I immeidately noticed that the model behaves more consistenly, making tuning easier. 

My method for tuning generally involved observing the validation accuracy while adjusting number of epochs and dropout rate. I remembered reading that Deep Learning is the art of regularization, so invested most of my effort in balancing improving training accuracy vs improving validation accuracy. I still wonder if increasing number of epochs would improve accuracy. I occasionally fidgetted with the learning rate when I felt that the accuracy started to oscillate.

I wish I had an NVIDIA GPU that would allow me to accelerate my tuning (in fact an NVIDIA-installed laptop's hard drive was messed trying to install a Linux partition), but found my end results satisfying but always room for improvements.

Note: I've reached test accuracy of at least 0.94 sometimes using a larger dataset (4000 examples per label) but found tuning horrendously painful to wait.
 
### Test Model on New Images

#### 1. Seven New Images

Here are six German traffic signs that I found on the web:

![alt text][image5]

The Stop sign and Do-not-enter sign image might be difficult to classify because there are many signs with red background with wide, white text or symbol.

Similar thing can be said about the Caution sign and Right-of-way-on-next sign but with a triangular shape.

The speed limit sign is particularly challenging because it has a red background and a numerical value which has be differentiated from other speed limit signs. 

#### 2. Model's Predictions

Here are the results of the prediction:

| Image                           |     Prediction                                | 
|:-------------------------------:|:---------------------------------------------:| 
| Stop Sign                       | Yield sign                                    | 
| Right-of-way @ next inter.      | Right-of-way @ next inter.                    |
| Priority                        | Priority                                      |
| 70 km/h                         | 70 km/h                                       |
| Roundabout                      | Roundabout                                    |
| Do not enter                    | Do not enter                                  |
| General Caution                 | General Caution                               |

The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 94.6%

#### 3. Analysis on Model's Predictions

This is a plot of the model's calculated classification probability for each web image.

![alt text][image6]

My model unfortunately predicted incorrectly on the first image, despite the correct label coming second in the probability. The fifth was surprsingly close. Otherwise, my model did generally well in classifying the signs. 

Top 5 Probabilities:

Prediction: 13, Actual: 14
    Label: 13, Value: 0.9425109028816223
    Label: 14, Value: 0.041424110531806946
    Label: 36, Value: 0.006646007299423218
    Label: 3, Value: 0.0029350263066589832
    Label: 1, Value: 0.002155170775949955

Prediction: 17, Actual: 17
    Label: 17, Value: 1.0
    Label: 14, Value: 1.9013152163793734e-11
    Label: 33, Value: 5.0927971150145634e-14
    Label: 13, Value: 4.941722690234539e-14
    Label: 9, Value: 5.564906937089451e-15

Prediction: 12, Actual: 12
    Label: 12, Value: 0.9998764991760254
    Label: 40, Value: 0.0001222667342517525
    Label: 3, Value: 6.838113222329412e-07
    Label: 38, Value: 2.540018897434493e-07
    Label: 32, Value: 2.2956331235945981e-07

Prediction: 11, Actual: 11
    Label: 11, Value: 1.0
    Label: 30, Value: 1.0661196370165271e-08
    Label: 27, Value: 1.1325663740180403e-09
    Label: 24, Value: 2.7971508442803383e-10
    Label: 18, Value: 2.3293100781529574e-10

Prediction: 18, Actual: 18
    Label: 18, Value: 0.5350461006164551
    Label: 38, Value: 0.3525824248790741
    Label: 1, Value: 0.06493526697158813
    Label: 2, Value: 0.024188155308365822
    Label: 39, Value: 0.009927839040756226

Prediction: 4, Actual: 4
    Label: 4, Value: 0.9860913157463074
    Label: 0, Value: 0.013442288152873516
    Label: 1, Value: 0.0004639170365408063
    Label: 8, Value: 2.5010699573613238e-06
    Label: 40, Value: 3.5614963422858636e-08

Prediction: 40, Actual: 40
    Label: 40, Value: 0.9913620352745056
    Label: 12, Value: 0.005916586145758629
    Label: 11, Value: 0.0025628714356571436
    Label: 1, Value: 0.00012677753693424165
    Label: 29, Value: 1.675482781138271e-05

