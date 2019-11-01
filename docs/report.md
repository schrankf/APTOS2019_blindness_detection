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

In this project, we try to achieve exactly this. We trainied a CNN to classify the degree of diabetic retinopathy based on real-world eye-screening images. The project is based on the kaggle competition [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/overview). The dataset comprising labeled eye-screening images is provided throughout kaggle.

### Problem Statement

The main objective of this project is to classify eye-screening fondus images with respect to the severity of diabetic retinopathy. The term fundus describes the interior surface of the eye. An example is given in figure 1. The provided dataset comprises 3662 labeled fundus images. 
The classification problem is non-binary, i.e., a multi-class problem. Labels are given categorical on a scale from 0 to 4 and describe the severity of the disease:

\begin{description}
\item[0] - No diabetic retinopathy
\item[1] - Mild
\item[2] - Moderate
\item[3] - Severe
\item[4] - Proliferative diabetic retinopathy
\end{description}

The dataset is governed by real-world variations such as image and photo artifacts and label noise such as misclassified disease state. It's worthwhile to mention, that the images are taken from multiple clinics and different (vendor) camera systems. Labels are given by multiple clinicians. Both should further increase variations. Images are provided as PNG in color RGB.

To solve this multi-class classification problem, I used a pre-trained ResNet50 model, initialized with weights from the [ImageNet](http://www.image-net.org/) dataset. Following the principles of transfer learning, I cut-off the top fully-connected layers and replace it with my own ones. The outcome layer holds 5 nodes, according to the 5 disease classes, yielding probabilities for each.

\begin{figure}
\centering
\includegraphics[width=0.6\textwidth]{../data/train_images/0a61bddab956.png}
\caption{Example of a fondus image from the dataset.}
\end{figure}

### Metrics

To evaluate the network's performance of this multi-class classification problem, I use the precision, recall, and cohen's kappa metric. For the definitions below, I will use TP for true positives, FP for false positives, TN for true negatives, and FN for false negatives.

Precision answers, what proportion of positive identifications are actually correct. It is defined as:
\begin{equation}
precision = \frac{TP}{TP + FP}
\end{equation}

Recall gives you what proportion of actual positives was identified correctly by:
\begin{equation}
recall = \frac{TP}{TP + FN}
\end{equation}

The third metric I used is Cohens kappa, defined as:
\begin{equation}
kappa = \frac{total~accuracy - random~accuracy}{1 - tandom~accuracy}
\end{equation}
where
\begin{equation}
total~accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}
and
\begin{equation}
random~accuracy = \frac{(TN + FP)\cdot(TN + FN) + (FN + TP)\cdot(FP + TP) }{total \cdot total}.
\end{equation}

Given this health-care domain, Recall is more critical than Precision, as it is essential to detect many diseases as possible [4]. 
For a multiclass problem, there are different ways to average the scores. I will report *macro* and *weighted* averaging. *Macro* will calculate the score for each class and take the mean of the scores. This will treat every class equally,  hence it is more honest for an imbalanced problem. *Weighted* will again calculate scores for each class, however, weight them concerning their support. By doing this, it is favoring the majority class. I will report both scores, however, for this imbalanced problem, *macro* seems to be the more meaningful [5]
I choose to add Cohen's kappa because it is the metric used in the original kaggle competition. Cohen's kappa tells you how much better your model performs compared to a random guess by chance [6]. Cohen's kappa is mostly used for the prediction based on unbalanced classes. I did not consider Accuracy as it is known to be a bad choice for imbalanced data (see section below). 

\pagebreak
## II. Analysis

### Data Exploration \& Exploratory Visualization

The dataset comprises 3662 colored PNG image files and a text file containing the filenames and labels. Five labels are given from 0 to 4. Images are provided in different shapes and sizes (see figure 2). The smallest images are given in height 358 by width 474, whereas the largest ones are given in height 2848 by width 4288 pixels. We will address this by rescaling every image to matching sizes. 
Figure 3 shows randomly picked examples from the dataset, covering every disease class. First, we can see that the images come in different crop-factors. More precisely, some of them display the complete circular fundus, whereas others are cropped at the top and bottom. Furthermore, the images differ in there brightness level. 
If we look carefully, there are more distinct features visible in the more severe disease stages. However, by the naked eye, it is hard to classify the severity of the disease.  

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{fig/resolution.png}
\caption{Distribution of the image sizes.}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=1.\textwidth]{fig/img1.png}
\includegraphics[width=1.\textwidth]{fig/img2.png}
\caption{Image samples covering every class.}
\end{figure}

One key observation is that the class distribution is highly imbalanced (see figure 4). Approximately 50% are class 0, i.e., healthy with no diabetic retinopathy. Class 1 (Mild) and 2 (Moderate) make up approx. 10% and 28%, respectively, while the severe stages 3 and 4 (Severe and Proliferative DR) make up under 10% each. 
Imbalanced data is usually critical for machine learning problems, e.g., as the classifier is more likely to classify an object to the majority class.
  
\begin{figure}[h]
\centering
\includegraphics[width=.8\textwidth]{fig/dist.png}
\caption{Imbalanced class distribution.}
\end{figure}

### Algorithms  and Techniques

For this project, I used a deep convolutional neural network. Convolutional neural networks (CNN) have been widely used and proven to be highly successful in image classification and recognition problems [7].

A CNN takes images as input, applies filters to extract specific features, and uses a deep neural network (i.e., fully connected layers) to classify the images based on the derived features. Typically features are, e.g., boundaries, colors, shape, and so on. The output of the fully connected layers is usually probabilities for each class, while the maximum probability will be used as the predicted class, i.e., disease stage.  

Because of my limited computing resources and the low number of images,  I used the pre-trained ResNet50 model following the approach of transfer learning. Here, I use already trained convolutional layers for feature extraction. The fully connected layers at the top that are used for classification are replaced by new ones, specific for the problem, e.g., classifies disease stages. 

### Benchmark
As a benchmark model I used a naive model approach. 
This model will predict every image to be classified as class 0, i.e., no diabetic retinopathy. The reason for this is that the majority of images (approx. 50%) are labeled as 0 (see figure 2). The results are computed by the metrics descriped above and given in table 1. The trained CNN should at least perfrom better than the this naive model.

| **Model** | **Average method** | **Precision** | **Recall** | **Cohens kappa** |
|-------|---------:|:------:|:------------:|:-------:|
| Naive class 0 | macro avg | 0.10 | 0.20 | |
| | weighted avg | 0.24 | 0.49 | |
| |  | | |  |
| |  | | | 0.0 |
Table: Results for the naive assumption.

## III. Methodology   

Implementation was done using [sci-kit learn](https://scikit-learn.org/) (version 0.21.3) and [keras](https://keras.io/) (version 2.2.4). Training was perfromed on a AWS EC2 instance and a Deep leanring AMI.

### Data Preprocessing
First, I used sci-kit learn's `train_test_split()` function to randomly shuffle and divide the dataset into 2334 train, 916 test, and 412 validation images according to 65%, 25%, 15%, respectively. The function was called twice, first, to split the dataset for training and testing by 75% / 25%, and second to split the training set ones more for validation by 85% / 15%. In both calls, the option `shuffle` was set to `True` with a fixed seed given as `random_state` equal to 42:

```python
train, test = train_test_split(df, test_size = 0.25, random_state = 42, shuffle = True)
train, val  = train_test_split(train, test_size = 0.15, random_state = 42, shuffle = True)
```

Second, to address the inconsistency on the provided images (see section II), every image needed to be resized to squared height 224 by width 244, rescaled to 4D (for Tensorflow as backend) and normalized by 255 (RGB). Furthermore, I used data augmentation to improve the performance of the model.
Here, the original train and validation images are replaced by ones that are randomly governed by 10% rotation, 10% zoom,  horizontal and vertical shifts by 0.1 and horizontal flips. Image rescaling, normalization, and the augmentation were implemented by using keras's `ImageDataGenerator`. 

I created first an `ImageDataGenerator()` object for training data, where I defined the parameters used for image augmentation and the normalization factor by 255. I choose to set values for rotation, zoom, width and height shift, and horizontal flips as given above:

```python
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
Hereafter, I use this generator to derive training and validation data generators using the `flow_from_dataframe()` method. This method takes the data frame `train` or `val`, comprising filenames (`x_col`) and labels (`y_col`), and the folder path, to generate batches of the size of 64 of the preprocessed and augmented data [REF]. It is worthwhile noting that the augmented data replacing the original data. The method also takes the new image size of 224 by 224, the color mode as RGB, and the corresponding classes:
```python
# setup
df_classes = ['0', '1', '2', '3', '4']
img_rescale = (224, 224)
batch_sz = 64

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

The reason to go with `flow_from_dataframe()` was that I was not able to load all images due to memory limits; importing all images had crashed my machine.
Therefore, I used `flow_from_dataframe()` as a memory friendly solution. 
Examples of augmented images are given in figure 5.

\begin{figure}[h]
\centering
\includegraphics[width= 1.\textwidth]{fig/augmentation.png}
\caption{Samples of augmented data used for training.}
\end{figure}


### Implementation

I used keras functional API to build the ResNet50 CNN. 
The ResNet50 model was imported with pre-trained weights from ImageNet and without the top layers, i.e., the fully connected layers for classification. I added my own top layers by deep neural network comprising `GlobalAveragePooling2D()`, `BatchNormalization()`, `Dropout()`, and `Dense()` layers. The complete model takes images with a shape of `(224,224,3)` as input and outputs the probabilities for all five classes. I set all weights to trainable. Overall the model has 28,843,909 total parameters. 
```python
# build pre-trained ResNet50 CNN
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

def get_resnet(img_shape, num_class): 
    input_img = Input(shape = img_shape)
    resnet50 = ResNet50(weights = 'imagenet', 
                        include_top = False, 
                        input_tensor = input_img)
    x = GlobalAveragePooling2D()(resnet50.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation = 'relu')(x)
    x = Dropout(0.4)(x)
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
The model was compiled using the `compile()` method with categorical cross-entropy for the loss function, stochastic gradient descent (`SGD`) with Nesterov accelerated momentum as the optimizer, and accuracy as a metric for evaluation.  

```python 
# compile the model
sgd = optimizers.SGD(nesterov = True)
adam = optimizers.Adam()

# compile
model_resnet.compile(loss = 'categorical_crossentropy', 
                     optimizer = sgd,
                     metrics = ['accuracy'])
```

I used three callback functions from Keras, which were applied during the training process. `ModelCheckpoint()` is used to save the best model to a given directory.  `EarlyStopping()` will stop training if the model is not improving after ten epochs. `ReduceLROnPlateau()` will reduce the learning rate of the optimizer if the learning stagnates. 

```python
# checkpointer
cb_checkpointer = ModelCheckpoint(filepath = 'models/model_resnet', 
                                  save_weights_only =  True,
                                  save_best_only = True,
                                  verbose = 1)
# early stopping
cb_earlystop = EarlyStopping(monitor = 'val_loss',
                             patience = 12,
                             verbose = 1)

# learning rate reduction
cb_learningrate = ReduceLROnPlateau(monitor = 'val_loss',
                                    patience = 3,
                                    factor = 0.1, 
                                    min_lr = 1e-6,
                                    verbose = 1)
```

Training is finally done using the `fit_generator()` method. This method takes the training and validation generators, the number of steps per epoch and validation, the callbacks, and the number of maximal epochs.

```python
# train model
model_resnet.fit_generator(
            generator = train_gen,
            steps_per_epoch = len(train_gen.filenames) // train_gen.batch_size,
            epochs = 40,
            validation_data = val_gen,
            validation_steps = len(val_gen.filenames) // val_gen.batch_size,
            callbacks = [cb_checkpointer, cb_early_stop, cb_learning_rate],
            workers = 4, # 4 for EC2
            use_multiprocessing = False)
            )
```

To evaluate the model's performance I used sci-kit learn's `classification_report()` and `cohen_kappa_score()` function. Both functions take true and predicted labels. In agreement with the original kaggle competition, I used quadratic weighting for Cohens kappa.

```python
# eval function
def eval_performance(y_true, y_pred):
    classes = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    report = classification_report(y_true, y_pred, target_names = classes)
    kappa = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')
    
    print(report)
    print('---------------------')
    print('Cohen''s Kappa:   {:1.2f}'.format(kappa))
```

### Refinement
To improve the model's performance, I tried various adjustments.
I realized that minor data augmentation works better, so I gradually decreased and removed parameters for the augmentation. 
I tried different top-layer architecture; however, it was hard to evaluate these due to the high computational costs to train several different models.
I noticed the prominent effect by setting the `Dropout()` layers to 40% and changing the activation function in the fully connected `Dense()` layers to `relu`.
I was surprised by how poor `adam` and `rmsProp` optimizers performed.  Using both, I actually never got the model to perform better than approximately  0.65 in terms of accuracy during training. Choosing `SGD` increased the model's performance significantly; this was quite a substantial performance boost. 
Adding the `ReduceLROnPlateau()` as callback function also increased the performance compare to the initial results. 

\pagebreak
## IV. Results

### Model Evaluation and Validation
Figure 6 shows the training history with training and validation learning curves. We can see that the loss curves are decreasing, while accuracies are increasing — the training convergences. After 32 epochs the early stopping callback interrupts the training. Based on the learning curves I cannot see evidence of overfitting. 

\begin{figure}[h]
\centering
\includegraphics[width= 1.\textwidth]{fig/learning.png}
\caption{Learning curves for training and validation.}
\end{figure}

The final CNN model achieved quite good results on the yet unseen testing data. The results for the test dataset are given in table 2.


| **Model** | **Class / Average** | **Support** |**Precision** | **Recall ** | **Cohens kappa** |
|-------|:---------|:----:|:------:|:------------:|:-------:|
|  Final CNN | 0 - No DR | 450 |0.93 | 0.98 | |
|  | 1 - Mild | 81 |0.58 | 0.57 |  |
|  | 2 - Moderate | 257 |0.74 | 0.75 |  |
|  | 3 - Severe | 44 |0.28 | 0.30 |  |
|  | 4 - Proliferative DR | 84 |0.59 | 0.39 | |
|  |  | |  | |
|  | Macro | 916 |0.63| 0.60 |   |
|  | Weighted | 916 |0.78 | 0.79 | |
|  |  | | | | |
|  |  | 916 | | | 0.86 |
Table: Results for the final CNN model on the unseen test data.

If we look carefully at the evaluation metrics given in table 2, we can see that the imbalanced of the data greatly affects the results. For example, the model achieved a precision of 0.94 for the largest group class 0 with a support of 450, while the smallest group class 5  with a support of 44 achieved only 0.32. 

The accuracy is comparable to the value during training with 0.79 and 0.84, respectively. The final model seems to generalize well. Given these results, I consider this model and the parameter as appropriate. 

### Justification
Table 3 compares both models, the naive assumption predicting every image to be class 0 (i.e., no DR), and the final CNN model. 

| **Model** | **Average method** | **Precision** | **Recall** | **Cohens kappa** |
|-------|---------:|:------:|:------------:|:-------:|
| Naive class 0 | Macro | 0.10 | 0.20 | |
|  | Weighted | 0.24 | 0.49 | |
|  | | | | 0.0 |
|  | | | |  |
| Final CNN | Macro | 0.63 | 0.60 | |
|  | Weighted | 0.78 | 0.79 | |
|  | | | | 0.86 |
Table: Model comparison of the naive and final CNN model.

The final CNN model outperforms the naive model by far. In particular, the critical macro averaged scores are way higher with 0.1 to 0.63 and 0.2 to 0.60 for precision and recall, respectively. Furthermore, Cohens kappa increased by zero to 0.86. 
However, to solve the problem fairly, I need more data. In particular, more data is necessary regarding higher disease stages, like class 4 and 5 (see section above and the imbalanced supports). 

\pagebreak
## V. Conclusion
### Free-Form Visualization

Figure 7 shows 20 randomly chosen images from the unseen test dataset. Each title displays the provided true and the predicted diagnosis. If the prediction is correct, then the font color is given green, else red.  We can see that 3 out of 20 images are correctly classified. 

\begin{figure}[h]
\centering
\includegraphics[width= 1.\textwidth]{fig/pred_1.png}
\includegraphics[width= 1.\textwidth]{fig/pred_2.png}
\includegraphics[width= 1.\textwidth]{fig/pred_3.png}
\includegraphics[width= 1.\textwidth]{fig/pred_3.png}
\caption{Samples of predicted images form the test dataset.}
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

Step 5 was a great moment. I realized that I couldn't load every image into memory and store it because my computer ran out of memory, and the python kernel crashed. Finally, I found the solution of using the `flow_from_dataframe()` method from `ImageDataGenerator` to import my images in batches during training.
Another great moment happened during step 7. In the first trails, I used the `adam` optimizer however, I couldn't get any good results during training. Only when I changed to `SGD` the training worked well and provided a significant performance boost. Furthermore, during the final training time, I finally managed to run the training on an AWS EC2 instance using the GPU. The performance boost was huge!

### Improvement
I think the given provided solution is decent and good; however, it can be improved. If we look carefully at the leaderboard on kaggle, the top-ranked notebooks use more elaborate image pre-processing. They adjust the color channels to highlight specific features, that may lead to an improved solution. Most of the techniques are implemented using fastai or PyTorch, while in this project, I wanted to focus on keras. Maybe one can increase the complexity of the CNN to achieve better results. However, this will also increase the computational coast.

### References

[1] Engelgau, Michael, Linda Geiss, Jinan Saaddine, Jame Boyle, Stephanie Benjamin, Edward Gregg, Edward Tierney, Nilka Rios-Burrows, Ali Mokdad, Earl Ford, Giuseppina Imperatore, K. M. Venkat Narayan. "The Evolving Diabetes Burden in the United States." Annals of Internal Medicine, 1 June 2004. Web. 22 Apr. 2014 

[2] https://nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy

[3] Caroline MacEwen. "diabetic retinopathy". Retrieved August 2, 2011.

[4] https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

[5] https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult

[6] https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english

[7] http://cs231n.github.io/convolutional-networks/#pool
