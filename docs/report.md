---
output:
  pdf_document: 
    highlight: pygments
  html_document: 
    highlight: pygments
---

\begin{center}
\huge{Machine Learning Engineer Nanodegree}

\Large{Capstone Project: Blindess detection}

\vspace{0.5cm}

\normalsize{\emph{Felix Schrank}} 

\emph{October 12th, 2019}
\end{center}

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
\item[2]-  Moderate
\item[3]- Severe
\item[4] - Proliferative diabetic retinopathy
\end{description}

The dataset is governed by real-world variations such as image and photo artifacts and label noise such as misclassified disease state. It's worthwhile to mention, that the images are taken from multiple clinics and different (vendor) camera systems. Labels are given by multiple clinicians. Both should further increase variations. Images are provided as PNG in color RGB.

To solve this multi-class classification problem, I used a pre-trained ResNet50 model, initialized with weights from the [ImageNet](http://www.image-net.org/) dataset. Following the principles of transfer learning, I cut-off the top fully-connected layers and replace it with my own ones. The outcome layer holds 5 nodes, according to the 5 disease classes, yielding probabilities for each.

\begin{figure}
\centering
\includegraphics[width=0.5\textwidth]{../data/train_images/0a09aa7356c0.png}
\caption{Example of a fondus image from the dataset.}
\end{figure}

### Metrics

To evaluate the network's performance of this multi-class classification problem, I use the precision, recall, and cohen's kappa metric. Precision answers, what proportion of positive identifications are actually correct. It is defined as:
\begin{equation}
Precision = \frac{TP}{TP + FP}
\end{equation}

In contrast, Recall gives you what proportion of actual positives was identified correctly by:
\begin{equation}
Recall = \frac{TP}{TP + FN}
\end{equation}

The third metric I used is Cohens kappa, defined as:
\begin{equation}
kappa = \frac{Total Accuracy - Random Accuracy}{1 -Random Accuracy }
\end{equation}
where
\begin{equation}
Total Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}
and
\begin{equation}
Random Accuracy = \frac{(TN + FP)\cdot(TN + FN) + (FN + TP)\cdot(FP + TP) }{Total \cdot Total}
\end{equation}

Given this health-care domain, Recall is more critical than Precision, as it is essential to detect many diseases as possible [4]. 
For a multiclass problem, there are different ways to average the scores. Two of which I will use are *macro* and *weighted*. *Macro* will calculate the score for each class and take the mean of the scores. This will treat every class equally,  hence is more critical for an imbalanced problem. *Weighted* will again calculate scores for each class, however, weight them concerning their support. By doing this, it is favoring the majority class. I will report both scores, however, for this imbalanced problem  *macro* seems to be the more meaningful [5]
I choose to add Cohen's kappa because it is the metric used in the original kaggle competition. Cohen's kappa tells you how much better your model performs, compared to a random guess by chance [6]. Cohen's kappa is mostly used for the prediction based on unbalanced classes. I did not consider Accuracy as it is known to be a bad choice for unbalanced data (see sections below). 

\pagebreak
## II. Analysis

### Data Exploration \& Exploratory Visualization

The dataset comprises 3662 colored PNG image files and a text file containing the filenames and labels. Five labels are given from 0 to 4. Images are provided in different shapes and sizes (see figure 2). The smallest images are given in height 358 by width 474 whereas the largest ones are given in height 2848 by width 4288 pixels. We will address this by rescaling every image to matching sizes. 
Figure 3 shows randomly picked examples from the dataset, covering every disease class. First, we can see that the images come in different crop-factors. More precisely, some of them display the complete circular fundus, whereas others are cropped at the top and bottom. Furthermore, the images differ in there brightness level. 
If we look carefully, there are more distinct features visible in the more severe disease stages. However, by the naked eye, it is hard to classify the severity of the disease.  

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{fig/resolution.png}
\caption{Distribution of the image sizes.}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=1.\textwidth]{fig/img1.png}
\includegraphics[width=1.\textwidth]{fig/img2.png}
\caption{Image samples of the data.}
\end{figure}

One key observation is that the class distribution is highly imbalanced (see figure 4). Approximately 50% are class 0, i.e., healthy with no diabetic retinopathy. Class 1 (Mild) and 2 (Moderate) make up approx. 10% and 28%, respectively, while the severe stages 3 and 4 (Severe and Proliferative DR) makes up under 10% each. 
Imbalanced data is usually critical for machine learning problems, e.g., as the classifier is more likely to classify an object to the majority class.
  
\begin{figure}[h]
\centering
\includegraphics[width=.7\textwidth]{fig/dist.png}
\caption{Imbalanced class distribution of the data.}
\end{figure}

### Algorithms  and Techniques

**TODO Batch size, epochs, ***

For this project, I used a deep convolutional neural network. Convolutional neural networks (CNN) have been widely used and proven to be highly successful in image classification and recognition problems [REF].

A CNN takes images as input, applies filters to extract specific features, and uses a deep neural network (i.e., fully connected layers) to classify the images based on the derived features. Typically features are, e.g., boundaries, colors, shape, and so on. The output of the fully connected layers is usually probabilities for each class, while the maximum probability will be used as the predicted class, i.e., disease stage.  

Because of my limited computing resources and the low number of images,  I used the pre-trained ResNet50 model following the approach of transfer learning. Here, I use already trained convolutional layers for feature extraction. The fully connected layers at the top that are used for classification are replaced by new ones, specific for the problem, e.g., classifies disease stages. 

### Benchmark
As a benchmark model I used a naive model approach. 
This model will predict every image to be classified as class 0, i.e., no diabetic retinopathy. The reason for this is that the majority of images (approx. 50%) are labeled as 0 (see figure 2). The results are computed by the metrics descriped above and given in table 1. The trained CNN should at least perfrom better than the this naive model.

| **Model** | **Average method** | **Precision** | **Recall** | **Cohens kappa** |
|-------|---------:|:------:|:------------:|:-------:|
| Naive class 0 | Macro avg | 0.10 | 0.20 | 0.0 |
| Naive class 0 | Weighted avg | 0.24 | 0.49 | 0.0 |
Table: Results for the naive assumption.

## III. Methodology 

Implementation was done using sci-kit learn (version 0.21.3) and Keras (version 2.2.4).

### Data Preprocessing
First, I used sklearn's `train_test_split()` function to randomly shuffle and divide the dataset into a 2334 train, 916 test, and 412 validation images according to 65%, 25%, 15%, respectively.
The function was called twice, first, to split the dataset for training and testing by 75% / 25%, and second to split the training set ones more for validation by 85% / 15%. 
In both calls, the option `shuffle` was set to `True` with a fixed seed given as `random_state` equal to 42:

```python
train, test = train_test_split(df, test_size = 0.25, random_state = 42, shuffle = True)
train, val  = train_test_split(train, test_size = 0.15, random_state = 42, shuffle = True)
```

Second, to address the inconsistency on the provided images (see section II), every image needed to be resized to squared height 224 by width 244, rescaled to 4D (for Tensorflow as backend), and normalized by 255 (RGB). Furthermore, I used data augmentation to improve the performance of the model.
Here, the original train and validation images are replaced by ones that are randomly goverend by 10% rotation, 10% zoom,  horizontal and vertical shifts by 0.1 and horizontal flips. Both, image rescaling and normailzation and the augmaentation was implemented by using kears's `ImageDataGenerator`. 

I created first an `ImageDataGenerator()` object for training data, where I defined the parameters used for image augmentation and the normalization factor by 255. I choose to set values for rotation, zoom, width and height shift, and horizontal flips as given above:

```python
# generator for training
train_generator = ImageDataGenerator(rescale = 1./255.,
                             fill_mode = 'constant',
                             cval = 0,
                             # featurewise_center = True, # subtract mean
                             # featurewise_std_normalization=True, # standardise
                             rotation_range = 10,
                             zoom_range = 0.3, 
                             shear_range = 0.,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             horizontal_flip = True)
```
Hereafter, I use this generator to derive training and validation data generators using the `flow_from_dataframe()` method. This method takes the data frame `train` or `val`, comprising filenames (`x_col`) and labels (`y_col`), and the folder path, to generate batches of the size of 32 of the preprocessed and augmented data [REF]. It is worthwhile noting that the augmented data replacing the original data. The method also takes the new image size of 224 by 224, the color mode as RGB, and the corresponding classes:
```python

# setup   
df_classes = ['0', '1', '2', '3', '4']
img_rescale = (224, 224)
batch_sz = 32

# data generator for training
train_gen = train_generator.flow_from_dataframe(dataframe = train,
                                              x_col = 'filename',
                                              y_col = 'diagnosis',
                                              directory = 'data/train_images/',
                                              target_size = img_rescale,  
                                              batch_size = batch_sz,
                                              color_mode = 'rgb',
                                              classes = df_classes)

#  data generator for validation
val_gen = train_generator.flow_from_dataframe(dataframe = val,
                                              x_col = 'filename',
                                              y_col = 'diagnosis',
                                              directory = 'data/train_images/',
                                              target_size = img_rescale,
                                              batch_size = batch_sz,
                                              color_mode = 'rgb',
                                              classes = df_classes)

```

The reason to go with `flow_from_dataframe()` was that I was not able to import all images due to memory limits; importing aall images had crashed my machine.
Therefore, I used `flow_from_dataframe()` as a memory friendly solution. 
Examples of augmented images are given in figure 5.

\begin{figure}[h]
\centering
\includegraphics[width= 1.\textwidth]{fig/augmentation.png}
\caption{Samples of augmented data for training.}
\end{figure}


### Implementation

I used Keras functional API to build the ResNet50 CNN. 
The ResNet50 model was imported with pre-trained weights from ImageNet and without the top layers, i.e., the fully connected layers for classification. I added my own top layers by deep neural network comprising `GlobalAveragePooling2D()`, `BatchNormalization()`, `Dropout()`,and `Dense()`. The complete model takes images with a shape of `(224,224,3)` as input and outputs the probabilities for all five classes. I set all weights to trainable. Overall the model has 25,957,765 total parameters. 
```python
# construct pre-trained ResNet50 Model
def get_resnet_model(img_shape, num_class): 
    input_img = Input(shape = img_shape)
    resnet50 = ResNet50(weights = 'imagenet', 
                        include_top = False, 
                        input_tensor = input_img)
    x = GlobalAveragePooling2D()(resnet50.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_class, activation = 'softmax')(x)
    return Model(input_img, output)

model_resnet = get_resnet_model((224,224,3), num_class = 5)
# setup trainable layers
for layers in model_resnet.layers:
    layers.trainable = True
```
The model was compiled using the `compile()` method with categorical cross-entropy for the loss function, stochastic gradient descent (`SGD` as the optimizer, and accuracy as a metric for evaluation.  

```python 
# compile model
model_resnet.compile(loss = 'categorical_crossentropy', 
                     optimizer = 'SGD',
                     metrics = ['accuracy'])
```

I used three callback functions from Keras, which are applied during the training process. `ModelCheckpoint()` is used to save the best model to a given directory.  `EarlyStopping()` will stop training if the model is not improving after ten epochs. `ReduceLROnPlateau()` will reduce the learning rate of the optimizer if the learning stagnates. 

```python
# Checkpointer
cb_checkpointer = ModelCheckpoint(filepath = 'models/model_resnet', 
                                  save_weights_only =  False,
                                  save_best_only = True,
                                  verbose = 1)
# Early stopping
cb_early_stop = EarlyStopping(monitor = 'val_loss',
                              patience = 10,
                              verbose = 1)

# Learning rate reduction
cb_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                                     patience = 4,
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
            callbacks = [cb_checkpointer, cb_early_stop, cb_learning_rate]
            )
```

To evaluate the model's performance I used sci-kit learn's `classification_report()` and `cohen_kappa_score()` function. Both functions take true and predicted labels. In agreement with the original kaggle competition, I used quadratic weighting for Cohens kappa.

```python
def eval_performance(y_true, y_pred):
    
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    report = classification_report(y_true, y_pred, target_names = classes)
    kappa = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')
    
    print('Cohen''s Kappa:   {:1.2f}'.format(kappa))
    print('------- Report ------- ')
    print(report)
```

### Refinement
To improve the model's performance, I tried various adjustments.
I realized that minor data augmentation works better, so I gradually decreased and/or removed parameters for the augmentation. 
I tried different top-layer architecture; however, it was hard to evaluate these due to the high computational costs to train several different models.
I noticed the biggest effect by setting the `Dropout` layers to 40% and changing the activation function in the fully connected layers to *relu*.
I was truly surprised by how poor adam and rmsProp optimizers performed.  Using both, I actually never got the model to perform greater than approx. 0.6 in accuracy during training. Choosing SDG increased the model's performance significantly; this was quite a tremendous performance boost. 
Adding the `ReduceLROnPlateau()` as callback function also increased the performance compare to the initial results. 

## IV. Results

### Model Evaluation and Validation
Figure 6 shows the training history with training and validation learning curves. We can see that the loss curves are decreasing, while accuracies are increasing; the training convergences. After epoch XX the early stopping callback interrupts the training. 

\begin{figure}
\centering
\includegraphics[width= 0.4\textwidth]{fig/augmentation.png}
\caption{Learning curves for training and validation.}
\end{figure}

The final model achieved quite promising results on the yet unseen testing data (see table XX). 


| **Model** | **Class / Average** | **Support** |**Precision** | **Recall** | **Cohens kappa** |
|-------|:---------|:----:|:------:|:------------:|:-------:|
|  final CNN | 0 - No DR | 450 |0.10 | 0.20 | |
|  | 1 - Mild | 81 |0.10 | 0.20 |  |
|  | 2 - Moderate | 257 |0.10 | 0.20 |  |
|  | 3 - Severe | 44 |0.10 | 0.20 | |
|  | 4 - Proliferative DR | 84 |0.10 | 0.20 | |
|  |  | |  | |
|  | Macro | 916 |0.10 | 0.20 |  |
|  | Weighted | 916 |0.10 | 0.20 | 0.0 |
Table: Results for the final CNN model on the unseen test data.


If we look carefully at the metrics given in table 2, we can see that the imbalanced of the data greatly affects the results. For example, the model achieved a precision of 0.94 for the largest group class 0 with a support of 450, while the smallest group class 5  with a support of 44 achieved only 0.32. 

The accuracy is comparable to the value during training with XX and XX, respectively. The final model seems to generalize well. Given these results, I consider this model and the parameter as appropriate. 

### Justification
Table XX compares both the naive assumption predicting every image as class 0, and the final CNN model. 

| **Model** | **Average method** | **Precision** | **Recall** | **Cohens kappa** |
|-------|---------:|:------:|:------------:|:-------:|
| Naive class 0 | Macro avg | 0.10 | 0.20 | 0.0 |
| Naive class 0 | Weighted avg | 0.24 | 0.49 | 0.0 |
| Final CNN | Macro avg | 0.10 | 0.20 | 0.0 |
| Final CNN | Weighted avg | 0.24 | 0.49 | 0.0 |
Table: Comparison.

While the CNN model outperforms the naive model to solve the problem truly, I need more data. In particular, more data is necessary regarding higher disease stages, like class 4 and 5. 

## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
ortant quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions In this section, you will need to provide some form of visualization that emphasizes an impto ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

\begin{figure}
\centering
\includegraphics[width= 0.4\textwidth]{fig/augmentation.png}
\caption{Examples of predicted data}
\end{figure}

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_


### References

[1] Engelgau, Michael, Linda Geiss, Jinan Saaddine, Jame Boyle, Stephanie Benjamin, Edward Gregg, Edward Tierney, Nilka Rios-Burrows, Ali Mokdad, Earl Ford, Giuseppina Imperatore, K. M. Venkat Narayan. "The Evolving Diabetes Burden in the United States." Annals of Internal Medicine, 1 June 2004. Web. 22 Apr. 2014 

[2] https://nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy

[3] Caroline MacEwen. "diabetic retinopathy". Retrieved August 2, 2011.

[4] https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2

[5] https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult

[6] https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english



-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?


