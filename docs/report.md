---
title: "Machine Learning Engineer Nanodegree"
subtitle: "Capstone Project: Blindess detection"
author: Felix Schrank
date: October 29, 2019
output:
  pdf_document: default
  html_document: default
urlcolor: blue  
---

## I. Definition

### Project Overview

Diabetic retinopathy (or diabetic eye disease) is the leading cause of blindness in working-age adults [1]. It is an eye condition in people who have diabetes in which it affects blood vessels in the retina, the innermost, light-sensitive layer of the tissue of the eye [2]. The disease affects a large number of people who have diabetes with increasing risk the longer a person has diabetes [3]. 

Diabetic retinopathy often not have early symptoms, hence annual eye screening is crucial and can reduce the number of people who develop vision-threatening retinopathy [2]. However, in rural and poor living areas, medical screening is challenging to manage because of the often limited healthcare support. These people genuinely benefit from an automated image-screening solution for early diagnosis. Deep learning, in particular, convolutional neural networks (CNN) are highly successful in such an image-based classification problem. A CNN-based diabetic retinopathy screening promises to deliver fast and accurate diagnosis, improve the accessibility and reducing the examination coast.

In this project, I try to achieve exactly this. I trainied a CNN to classify the degree of diabetic retinopathy based on real-world eye-screening images. The project is based on the kaggle competition [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/overview). The dataset contains labeled eye-screening images is provided throughout kaggle.

### Problem Statement

The main objective of this project is to classify eye-screening fondus images with respect to the severity of diabetic retinopathy. The term fundus describes the interior surface of the eye. An example is given in figure 1. The provided dataset comprises 3662 labeled fundus images. 
The classification problem is non-binary, i.e., multi-class. Labels are given categorical on a scale from 0 to 4 and describe the severity of the disease:

\begin{description}
\item[0] - No diabetic retinopathy
\item[1] - Mild
\item[2] - Moderate
\item[3] - Severe
\item[4] - Proliferative diabetic retinopathy
\end{description}

The dataset is governed by real-world variations such as image and photo artifacts and label, e.g., as misclassified disease state. It's worthwhile to mention, that the images are taken from multiple clinics and different (vendor) camera systems. Images are labeled by various clinicians. Both should further increase variations. Images are provided as PNG in color RGB.

To solve this multi-class classification problem, I used a pre-trained ResNet50 model, initialized with weights from the [ImageNet](http://www.image-net.org/) dataset. Following the principles of transfer learning, I cut-off the top fully-connected layers and replace it with my new problem-specific ones. The outcome layer features 5 nodes, corresponding to the 5 disease classes, yielding each class's probabilities.

\begin{figure}
\centering
\includegraphics[width=0.6\textwidth]{../data/train_images/0a61bddab956.png}
\caption{Example of a fondus image from the dataset.}
\end{figure}

### Metrics

To evaluate the network's performance of this multi-class classification problem, I use the precision, recall, and cohen's kappa metric. For the definitions below, I will use TP for true positives, FP for false positives, TN for true negatives, and FN for false negatives.

Precision answers, what proportion of positive identifications are actually correct. It is defined as
\begin{equation}
precision = \frac{TP}{TP + FP}
\end{equation}

Recall answers, what proportion of actual positives was identified correctly by
\begin{equation}
recall = \frac{TP}{TP + FN}
\end{equation}

The third metric I used is cohen's kappa, defined as
\begin{equation}
kappa = \frac{total~accuracy - random~accuracy}{1 - random~accuracy}
\end{equation}
where
\begin{equation}
total~accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}
and
\begin{equation}
random~accuracy = \frac{(TN + FP)\cdot(TN + FN) + (FN + TP)\cdot(FP + TP) }{total \cdot total}.
\end{equation}

Given this health-care domain, recall is more critical than precision, as it is essential to detect many diseases as possible [4]. 
For a multiclass problem, there are different ways to average the scores. I will report *macro* and *weighted* averaging. *Macro* will calculate the score for each class and take the mean of the scores. This will treat every class equally,  hence it is more honest for an imbalanced problem. *Weighted* will again calculate scores for each class, however, weight them concerning their support. By doing this, it is favoring the majority class. I will report both scores, however, for this imbalanced problem, *macro* seems to be the more meaningful [5].
I choose to add cohen's kappa because it is the metric used in the original kaggle competition. Cohen's kappa tells you how much better your model performs compared to a random guess by chance [6]. Cohen's kappa is mostly used for the prediction based on imbalanced classes. 

I did not consider accuracy as it is known to be a bad choice for imbalanced data (see section below). 

\pagebreak
## II. Analysis

### Data Exploration \& Exploratory Visualization

The dataset comprises 3662 colored PNG image files and a flat text file containing the filenames and corresponding class labels. Five labels are given from 0 to 4. Images are provided in different shapes and sizes (see figure 2). The smallest images are given in height 358 by width 474, whereas the largest ones are given in height 2848 by width 4288 pixels. I will address this by rescaling every image to matching sizes. 

\begin{figure}[h!]
\centering
\includegraphics[width=0.6\textwidth]{fig/resolution.png}
\caption{Distribution of the image sizes.}
\end{figure}

Figure 3 shows randomly picked sample images from the dataset, covering each disease class. First, we can see that the images come in different crop-factors. More precisely, some of them display the complete circular fundus, whereas others are cropped at the top and bottom. Furthermore, the images differ in there brightness level. 
If we look carefully, there are more distinct features visible in the more severe disease stages. However, by the naked eye, it is hard to classify the severity of the disease.  

\begin{figure}[h!]
\centering
\includegraphics[width=.9\textwidth]{fig/img1.png}
\includegraphics[width=.9\textwidth]{fig/img2.png}
\includegraphics[width=.9\textwidth]{fig/img3.png}
\caption{Image samples covering each class 0 to 4.}
\end{figure}

One key observation is that the class distribution is highly imbalanced (see figure 4). Approximately 50% are class 0, i.e., healthy with no diabetic retinopathy. Class 1 (Mild) and class 2 (Moderate) make up approx. 10% and 28%, respectively, while the severe stages 3 and 4 (Severe and Proliferative DR) make up under 10% each. 
Imbalanced data is usually critical for machine learning problems, e.g., as the classifier is more likely to classify an image correctly from the majority class.
  
\begin{figure}[h!]
\centering
\includegraphics[width=0.7\textwidth]{fig/dist.png}
\caption{Imbalanced class distribution.}
\end{figure}

### Algorithms  and Techniques

For this project, I used a deep convolutional neural network (CNN). 
Classical supervised learning algorithms, such as multi-layer perception (MLP), assume 1-dimensional input data. Hence, images have to vectorized for classification and thereby losing any spatial pieces of information. In contrast, CNN makes the explicit assumption that the inputs are images in 3 dimensions (width, height, and depth) - preserving any spatial information. CNNs have been widely used and proven to be highly successful in image classification and recognition problems [7]. Generally speaking, CNN takes images as input, applies different filters to extract image features (e.g., edges, shapes, colors, etc.), and use a deep neural network for classification based on the derived features. The main building blocks of the feature extraction part are convolutional layers and pooling layers and for the classification part fully connected layers (see figure 5). 

\begin{figure}[h!]
\centering
\includegraphics[width=.7\textwidth]{fig/cnn.png}
\caption{Overview of a basic CNN. Source \href{https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html}{https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html}}
\end{figure}


Convolutional layers are a set of spatial filters (or kernels) that are only partially connected to the image [7]. In the so-called convolutional process, each filter window slides across the width and height of the image volume (in 3D), compute a dot product between the image values and the filter's weights, and produce is a 2d activation map. The filter's weights and bias values are trainable parameters.  Activation maps can extract, e.g., edges, colors, shapes, and so on. The output of convolutional layers are multiple 2d activation maps that are stacked on top and increase the depth dimension corresponding to the number of applied filters. Important hyperparameters are the filter window size (or kernel size) in pixel, the number of applied filters in each layer, and stride and padding values, which define the step size of the moving filter and how the filter handle missing values at the image edges, respectively. An illustration is given in figure 6. While convolutional layers increase the depth dimension (corresponding to the number of filters), we use pooling layers is to decrease the width and height dimension to reduce the number of parameters [8].  A 2-dimensional filter with given window size slides across the image and pool the entries. Most commonly used filters compute the maximum or the mean of the values inside the filter window. Pooling layers reduce overfitting and computational costs. An example of a max polling layer is given in figure 7.

\begin{figure}[h]
\centering
\includegraphics[width=.6\textwidth]{fig/conv.png}
\caption{Illustraion of the convolutional filter. Source \href{https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/}{https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/}}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=.5\textwidth]{fig/pool.png}
\caption{Rxample of a maximum polling layer. Source \href{https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2}{https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2}}
\end{figure}

After the feature extraction, we use a more classical deep neural network for classification. The main building blocks are fully connected layers that are fully linked to the 1-dimensional input. The main hyperparameters are the number of nodes and the activation function, such as *relu* for non-linearity. The last fully connected layer is the output layer with the number of nodes corresponding to the number classes, each assigns the class probabilities. I will use the maximum probability to be the predicted class. In the project's implementation, I further add dropout layers to the classification network. Dropout layers partially deactivate a given percentage of nodes to force all nodes to learn. This function helps to improve generalization and prevent overfitting [9].

I used a pre-trained ResNet50 model following the approach of transfer learning. Here, I import the feature extraction part of the model ResNet50 with already trained weights based on the large amount of images from the ImageNet dataset. On top, I build a new deep neural network for classification with the last output layer assigning probabilities to the five disease stage. Transfer learning is a great way to re-use a model (architecture and/or weights) as a starting point for a new problem. Benefits are, e.g., faster learning and convergence. In this project, I also benefit because of the limited number of labeled images. 

The ResNet50 model was used because it is one of the most popular used architecture for image classification. ResNet50 uses skip connections to overcome the vanish gradient problem of (very) deep architectures by propagating information over layers. The number 50 defines the number of layers [10].


### Benchmark
As a benchmark model I used a naive model approach. 
This model will predict every image to be classified as class 0, i.e., no diabetic retinopathy. The reason for this is that the majority of images (approx. 50%) are labeled as 0 (see figure 3 above). The results are computed by the metrics descriped above and provided in table 1. The trained CNN should at least perfrom better than the this naive model.

| **Model** | **Average method** | **Precision** | **Recall** | **Cohens kappa** |
|-------|---------:|:------:|:------------:|:-------:|
| Naive class 0 | macro avg | 0.10 | 0.20 | |
| | weighted avg | 0.24 | 0.49 | |
| |  | | |  |
| |  | | | 0.0 |
Table: Results for the naive assumption.

\pagebreak
## III. Methodology   

Implementation was done using [sci-kit learn](https://scikit-learn.org/) (version 0.21.3) and [keras](https://keras.io/) (version 2.2.4). Training was perfromed on a AWS EC2 instance utilizing a Ubuntu deep leanring AMI.

### Data Preprocessing
I used sci-kit learn's `train_test_split()` function to randomly shuffle and divide the dataset into 2334 train, 916 test, and 412 validation images according to 65%, 25%, 15%, respectively. The function was called twice, first, to split the dataset for training and testing by 75% / 25%, and second to split the training set ones more for validation by 85% / 15%. In both calls, the option `shuffle` was set to `True` with a fixed seed given as `random_state` equal to 42:

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.25, random_state = 42, shuffle = True)
train, val  = train_test_split(train, test_size = 0.15, random_state = 42, shuffle = True)
```

Next, to address the inconsistency of the provided images (see section II), every image needed to be resized to squared height 224 by width 244, rescaled to 4D (for Tensorflow as backend) and normalized by 255 (for RGB). Data augmentation was used to improve the performance of the model. Here, the original train and validation images are replaced by ones that are randomly governed by 10% rotation, 10% zoom,  horizontal and vertical shifts by 0.1 and horizontal flips. Image rescaling, normalization, and the augmentation were implemented by using keras's `ImageDataGenerator()`. 

First, I created an `ImageDataGenerator()`-object for the training data, where I defined the parameters used for image augmentation and normalization:

```python
from keras.preprocessing.image import ImageDataGenerator

# generator for training
train_generator = ImageDataGenerator(rescale = 1./255.,
                                     fill_mode = 'constant',
                                     cval = 0,
                                     rotation_range = 10,
                                     zoom_range = 0.3, 
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     horizontal_flip = True)
```
Hereafter, I use this generator to derive training and validation data generators using the `flow_from_dataframe()` method. This method takes the dataframe `train` or `val`, comprising a filename (`x_col`) and label (`y_col`) column, and the folder path, to generate batches of the size of 32 of the preprocessed and augmented data. The method also takes as input the the new image size of 224 by 224, the color mode as 'rgb', and the corresponding class labels:
```python
# setup
df_classes = ['0', '1', '2', '3', '4']
img_rescale = (224, 224)
batch_sz = 32

# generator for train
train_gen = train_generator.flow_from_dataframe(dataframe = train,
                                                x_col = 'filename',
                                                y_col = 'diagnosis',
                                                directory = 'data/train_images/',
                                                target_size = img_rescale,
                                                batch_size = batch_sz,
                                                color_mode = 'rgb',
                                                classes = df_classes)

# generator for validation
val_gen = train_generator.flow_from_dataframe(dataframe = val,
                                              x_col = 'filename',
                                              y_col = 'diagnosis',
                                              directory = 'data/train_images/',
                                              target_size = img_rescale,
                                              batch_size = batch_sz,
                                              color_mode = 'rgb',
                                              classes = df_classes)
```

The reason to use `flow_from_dataframe()` was that I was not able to load all images at once due to memory limits; importing all images had crashed my machine. Therefore, I used `flow_from_dataframe()` as a memory friendly solution. 
Examples of derived augmented images from the train generator are given in figure 8.

\begin{figure}[h]
\centering
\includegraphics[width= 1.\textwidth]{fig/augmentation.png}
\caption{Samples of augmented data used for training.}
\end{figure}


### Implementation

I used keras functional API to build the CNN re-using a pre-trained ResNet50 model. I imported the ResNet50 model with pre-initialized weights from the ImageNet dataset and without the top layers, i.e., the fully connected layers. On top, I added a deep neural network for the classification. This comprises a `GlobalAveragePooling2D()` layer for pooling the 3-dimensional into 1-dimensional data, a fully connected `Dense()` layer with 2048 nodes, and an output `Dense()` layer with five nodes, corresponding to the five disease stages, and a `softmax` activation function. The complete model takes images with a shape of `(224,224,3)` as input and outputs the class's probabilities. I set all layers to trainable. 

```python
# build pre-trained ResNet50 CNN
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense

def get_resnet(img_shape, num_class): 
    input_img = Input(shape = img_shape)
    resnet50 = ResNet50(weights = 'imagenet', 
                        include_top = False, 
                        input_tensor = input_img)
    x = GlobalAveragePooling2D()(resnet50.output)
    x = Dense(2048)(x)
    output = Dense(num_class, activation = 'softmax')(x)
    return Model(input_img, output)
# build model
model_resnet = get_resnet((224,224,3), num_class = 5)
# setup trainable layers
for layers in model_resnet.layers:
    layers.trainable = True
```
The model was compiled using the `compile()` method of the model object. I used categorical cross-entropy as the loss function, stochastic gradient descent (`SGD`) as the optimization algorithm, and `accuracy` as a metric to eval the model's performance during the training.

```python 
from keras import optimizers

# compile the model
sgd = optimizers.SGD()

# compile
model_resnet.compile(loss = 'categorical_crossentropy', 
                     optimizer = sgd,
                     metrics = ['accuracy'])
```

I also implemented callback functions from keras. Callback functions are applied during the training process. `ModelCheckpoint()` is used to save the best model's weights to a given directory.  `EarlyStopping()` is used used to stop the training if the model's performance (based on the validation loss) is not improving after 12 epochs.

```python
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# checkpointer
cb_checkpointer = ModelCheckpoint(filepath = 'models/model_resnet', 
                                  save_weights_only =  True,
                                  save_best_only = True,
                                  verbose = 1)
# early stopping
cb_earlystop = EarlyStopping(monitor = 'val_loss',
                             patience = 12,
                             verbose = 1)

```

To train the built and compiled model, I use the `fit_generator()` method. This method takes as input the training and validation generators, the number of steps per epoch and validation, the callback functions, and the number of maximal epochs. Furthermore, I defined `workers` to 4 if I run the training on the AWS EC2 instance using CUDA. 

```python
# train model
model_resnet.fit_generator(
            generator = train_gen,
            steps_per_epoch = len(train_gen.filenames) // train_gen.batch_size,
            epochs = 40,
            validation_data = val_gen,
            validation_steps = len(val_gen.filenames) // val_gen.batch_size,
            callbacks = [cb_checkpointer, cb_early_stop],
            workers = 4,
            use_multiprocessing = False)
            )
```
This initial model was successfully trained. However, the displayed learning curves indicate overfitting (see figure 9). The training loss is decreasing, and training accuracy is increasing, while both validation curves did not improve after approximately five epochs. Even worst, the validation loss and accuracy seem to increase and decrease, respectively, after 5 epochs.

\begin{figure}[h!]
\centering
\includegraphics[width= 1.\textwidth]{fig/initial.png}
\caption{Learning curves of the intital model.}
\end{figure}

To evaluate the model's performance I used sci-kit learn's `classification_report()` and `cohen_kappa_score()`. Both functions take true and predicted labels as input. In agreement with the original kaggle competition, I used quadratic weighting for cohen's kappa.

```python
from sklearn.metrics import cohen_kappa_score, classification_report

# eval function
def eval_performance(y_true, y_pred):
    classes = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    report = classification_report(y_true, y_pred, target_names = classes)
    kappa = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')
    
    print(report)
    print('---------------------')
    print('Cohen''s Kappa:   {:1.2f}'.format(kappa))
```

During the implementation of the code I did not encounter complications or diffuculties. 

### Refinement
The initial model defined in the section above was modified to improve the model's performance and especially prevent the overfitting.
First, I modified the deep neural network for classification by adding hidden layers. I added another fully connected `Dense` layer with 512 nodes. Additionally, both `Dense()` layers ńow uses the `relu` as nonlinear activation function. To address the overfitting, three `Dropout` layers with rates of 20%, 20%, and 40%, respectively. The final model was:

```python
# build pre-trained ResNet50 CNN
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense

def get_resnet(img_shape, num_class): 
    input_img = Input(shape = img_shape)
    resnet50 = ResNet50(weights = 'imagenet', 
                        include_top = False, 
                        input_tensor = input_img)
    x = GlobalAveragePooling2D()(resnet50.output)
    x = Dropout(0.2)(x)
    x = Dense(2048, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_class, activation = 'softmax')(x)
    return Model(input_img, output)
# build model
model_resnet = get_resnet((224,224,3), num_class = 5)
# setup trainable layers
for layers in model_resnet.layers:
    layers.trainable = True
```
In addition to this, I added the option for nesterov accelerated gradient in the `SGD()` optimizer, for more robust convergence. Changing the optimizer from `SGD` to `adam` or `rmsProp` did not improved the performance; actually quite the opposite.
```python 
# compile the model
sgd = optimizers.SGD(nesterov = True)

# compile
model_resnet.compile(loss = 'categorical_crossentropy', 
                     optimizer = sgd,
                     metrics = ['accuracy'])
```

Last, I added another callback function `ReduceLROnPlateau()`, which will reduce the learning rate of the optimizer if the learning stagnates.
````python
# learning rate 
cb_lr = ReduceLROnPlateau(monitor = 'val_loss',
                         patience = 3,
                         factor = 0.1,
                         min_lr = 1r-6,
                         verbose = 1)
````

\pagebreak
## IV. Results

### Model Evaluation and Validation
Figure 10 shows the training history with training and validation learning curves. We can see that the loss curves are decreasing, while accuracies are increasing — the training convergences. Based on the learning curves I cannot see evidence of overfitting. 

\begin{figure}[h]
\centering
\includegraphics[width= 1.\textwidth]{fig/final_lr2.png}
\caption{Learning curves for training and validation.}
\end{figure}

The final CNN model achieved quite good results on the yet unseen testing data. The results for the test dataset are given in table 2.


| **Model** | **Class / Average** | **Support** |**Precision** | **Recall ** | **Cohens kappa** |
|-------|:---------|:----:|:------:|:------------:|:-------:|
|  Final CNN | 0 - No DR | 450 |0.95 | 0.98 | |
|  | 1 - Mild | 81 |0.58 | 0.38 |  |
|  | 2 - Moderate | 257 |0.72 | 0.77|  |
|  | 3 - Severe | 44 |0.36 | 0.48 |  |
|  | 4 - Proliferative DR | 84 |0.54 | 0.42 | |
|  |  | |  | |
|  | Macro | 916 |0.63| 0.61 |   |
|  | Weighted | 916 |0.79 | 0.79 | |
|  |  | | | | |
|  |  | 916 | | | 0.87 |
Table: Results for the final CNN model on the unseen test data.

If we look carefully at the evaluation metrics given in table 2, we can see that the imbalance of the data greatly affects the results. For example, the model achieved a precision of 0.95 for the class 0 with the largest support of 450, while the class 3 with a small support of 44 achieved only 0.36. 

The accuracy is comparable to the value during training with 0.79. The final model seems to generalize well. Given these results, I consider this model and the parameter as appropriate. 

### Justification
Table 3 compares both models, the naive assumption predicting every image to be class 0 (i.e., no DR), and the final CNN model. 

| **Model** | **Average method** | **Precision** | **Recall** | **Cohens kappa** |
|-------|---------:|:------:|:------------:|:-------:|
| Naive class 0 | Macro | 0.10 | 0.20 | |
|  | Weighted | 0.24 | 0.49 | |
|  | | | | 0.0 |
|  | | | |  |
| Final CNN | Macro | 0.63 | 0.61 | |
|  | Weighted | 0.79 | 0.79 | |
|  | | | | 0.87 |
Table: Model comparison of the naive and final CNN model.

The final CNN model outperforms the naive model by far. In particular, the critical *macro*-averaged scores are higher with 0.1 to 0.63 and 0.2 to 0.61 for precision and recall, respectively. Furthermore, cohen's kappa increased from zero to 0.87. 
However, to solve this problem really good, I need more data. In particular, more data is necessary regarding higher disease stages, like class 4 and 5 (see section above and the imbalanced supports). 

\pagebreak
## V. Conclusion
### Free-Form Visualization

Figure 11 shows 20 randomly chosen images from the unseen test dataset covering each class. Each title displays the provided true and the predicted diagnosis. If the prediction is correct, then the font color is given green, else red. We can see that 4 out of 20 sample-images are not correctly classified. Every class 0 is correctly classified, while 2 out of 4 class 4 is not. Given the sample images, I have the feeling that the trained classifier faces problems to classify under-exposed images. On the other hand, the classifiers seem to work well in good exposed and clear images. This raises the idea to put more effort into the preprocessing.

\begin{figure}[h!]
\centering
\includegraphics[width= 1.\textwidth]{fig/pred_1.png}
\includegraphics[width= 1.\textwidth]{fig/pred_2.png}
\includegraphics[width= 1.\textwidth]{fig/pred_3.png}
\includegraphics[width= 1.\textwidth]{fig/pred_4.png}
\caption{Samples of predicted images form the test dataset.}
\end{figure}

Figure 12 shows the distribution of predictions of each (true) class. 
The most interesting finding in this figure is that the classifier separates healthy subjects and subjects who suffer from a serve disease stage quite well. For example, if we look at the distribution for prediction of class 3, class 4, only minor subjects are incorrectly classified as healthy. The prediction spread is majorly around high disease stages.
In contrast, the classifier is very robust to classify healthy subjects as healthy.
A challenge for classifier seems to be class 1, with quite a large spread between class 1 and class 2 predictions. However, we have to keep in mind that the support of class 1 was very little.

\begin{figure}[h!]
\centering
\includegraphics[width= 1.\textwidth]{fig/pred_dist2.png}
\caption{Distribution of predictions of each (true) class.}
\end{figure}

### Reflection
This project aimed to classify fondus images with respect to the severity of diabetic retinopathy.  Therefore, I trained a convolutional neural network that takes the images and predicts the stage of the disease. The process can be summarized by:

1. I downloaded the image dataset from [kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data).
2. The label file containing all filenames and their labels was imported.
3. Explorative data analysis was performed, obtaining different image sizes, shapes, and imbalanced class distribution.
4. I divided the images into train, test, and validation datasets.
5. A train and validation data generator was created, which handles image rescaling, normalization, and augmentation. 
6. A Benchmark model was created (naive assumption)
7. I created the CNN based on a pre-trained ResNet50 model, including callback functions.
8. The CNN was compiled and trained
9. Model's performance was evaluated

Step 5 was a great moment. I realized that I couldn't load every image into memory and store it because my computer ran out of memory, and the python kernel crashed. Finally, I found the solution of using the `flow_from_dataframe()` method from `ImageDataGenerator` to import my images in batches during training. Another great moment was during the final training time in step 8, where I finally managed to run the training on an AWS EC2 instance using the GPU. The performance boost was huge!

### Improvement
I think the given provided solution is decent and good; however, it can be improved. If we look carefully at the leaderboard on kaggle, the top-ranked notebooks use more elaborate image pre-processing. They adjust the color channels to highlight specific features, that may lead to an improved solution. Most of the techniques are implemented using fastai or PyTorch. Maybe one can increase the complexity of the CNN to achieve better results. However, this will also increase the computational coast.

### References

[1] Engelgau, Michael, Linda Geiss, Jinan Saaddine, Jame Boyle, Stephanie Benjamin, Edward Gregg, Edward Tierney, Nilka Rios-Burrows, Ali Mokdad, Earl Ford, Giuseppina Imperatore, K. M. Venkat Narayan. "The Evolving Diabetes Burden in the United States." Annals of Internal Medicine, 1 June 2004. Web. 22 Apr. 2014 

[2] https://nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy

[3] Caroline MacEwen. "diabetic retinopathy". Retrieved August 2, 2011.

[4] https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

[5] https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult

[6] https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english

[7] http://cs231n.github.io/convolutional-networks/#pool

[8] https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/

[9] https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/dropout_layer.html

[10] https://arxiv.org/abs/1512.03385