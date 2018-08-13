<p style="font-weight:bold;font-size:16pt" align="center">Classification on CIFAR-10 Using Machine Learning and Deep Learning Methods</p>

<p style="font-style:italic;font-size:12pt" align="center">Author: Kunze Wang, Dong Lu, Jiayu Gong</p>

## Abstract
*In this project, we are going to do classification on CIFAR-10 image dataset. The dataset has 60,000 images in total with 10 classes. Image classification is one of the essential and most popular problems in computer vision. It is widely used in a diversity of areas such as spectrum analysis, public security, and automation systems. We use 2 preprocessing methods including PCA and K means feature learning and extraction and 5 machine learning and deep learning models including KNN, Naive Bayes, Random Forest, Multilayer Perceptron, and CNN. Among these methods, CNN has the highest accuracy but takes the longest time. Other methods are fast but only can achieve an accuracy of about 50%.*



## 1. Introduction

Nowadays, image classification is more and more important in many different fields. For image search engine, after crawling the data from the websites, although there is “alt” attribute telling what the picture is about, the search engine also need to find another way to classify the pictures automatically. For some smartphone camera applications, they need to distinguish faces from other things. For auto-driving cars, it is essential for these cars to distinguish the objects in front of them.

In this project, we are going to do classification on CIFAR-10 image dataset. CIFAR-10 is a well-known image dataset, which contains 60000 32x32 colour-images in 10 classes, with 6000 images per class. We are going to use different machine learning and deep learning classifiers on this dataset, and analyze the accuracy and execution statistics. The methods we are going to use are KNN, Naive Bayes, Random Forest and  Convolutional Neural Networks.

CIFAR-10 human accuracy is approximately 94%$^1$. With some deep learning methods, the accuracy can achieve 96.53%$^2$, which is even above the human accuracy. So doing machine learning and deep learning on image classification not only saves us a lot of time by automatically classifying the images, but can also improve the accuracy.



## 2. Previous work

Deep learning methods, especially convolutional neural networks (CNNs), are extraordinarily popular used in previous work of CIFAR-10. 

Benjamin Graham$^2$ et al. focused on adjusting max-pooling parameter alpha when doing CNN. They tried to assign a non-integer value to alpha, while the traditional ways only allowed alpha to take an integer multiplicative value. Their form of fractional max-pooling reduced overfitting on a variety of datasets even without dropout.

Jost Tobias Springenberg$^3$ et al. were also optimizing the CNN method concentrating on modifying max-pooling procedure. They engaged in replacing max-pooling by a convolutional layer with increased stride without loss in accuracy on several image recognition benchmarks. A new deconvolution approach variant has been introduced for visualizing features learned by CNNs, which can be applied to a broader range of network structures than existing approaches.

In addition to using neural networks, there is some previous work applying pure machine learning algorithms. Jasper Snoek$^4$ et al. used a Bayesian optimization with Gaussian process priors. Their algorithm took the variable cost (duration) into consideration and it can leverage the presence of multiple cores for parallel experimentation algorithms. They improved previous automatic procedures and made it possible to reach or surpass human expert-level optimization for many algorithms including latent Dirichlet allocation, structured SVMs and convolutional neural networks.

There are works aiming at optimizing the data preprocessing step as well. Adam Coates$^5$ et al. did a detailed analysis of the effect of changes in the neural network model setup. It is shown that, besides the large numbers of hidden node, dense feature extraction is also critical to outstanding performance.



## 3. Methods

### 3.1 Preprocessing

#### 3.1.1 Grayscale

Grayscale is a common preprocessing method when doing machine learning on images. For CIFAR-10 dataset, every image has $32 \times 32\times 3$. $32\times32$ means the size of the image, and 3 means the image is composed of three independent channels for red, green and blue. The color image can be converted to grayscale by simply calculate $w_1v_{red}+w_2 v_{green}+w_3v_{blue}$. There are 2 combinations of W, one is $[0.299, 0.587, 0.114]$, the other is $[0.2126, 0.7152, 0.0722]$. This is a very common practice when dealing with handwriting image classification, for example, MNIST, because after transforming to grayscale, the features will rely more on shapes, not colours. 

However, for this project, grayscale is not a good choice. For example, many “plane” images’ background is the sky, which is blue. Many frogs are green. These colours are also the features of the image. Actually, for this project, after transforming the colour image to grayscale image, the accuracy will decrease for about 8%.

#### 3.1.2 Standard Score

Before doing feature learning such as PCA, it is quite important to calculate standard score. Standard score is kind of normalizing by subtracting the mean and then dividing the difference by the standard deviation.  The calculation process is $z = \frac{x-\mu}{\sigma}$, $\mu$ is the mean of the population. $\sigma$ is the standard deviation of the population.

Calculating standard score only changes the raw data’s mean to 0 and standard deviation to 1. The benefit of doing “minus mean” is to make PCA robust. When the coordinate changes, subtracting the mean value will always makes the direction of PCA not change. By dividing by the standard deviation before PCA, which is one kind of normalization, the PCA result won’t be influenced by the variance of the data. For this project, calculating standard score will increase the accuracy for about 1%.

#### 3.1.3 PCA

Principal Component Analysis (PCA) is a statistical method which is transforming a set of probably correlated variables into a set of linearly uncorrelated variables which are called principal components. When a dataset consists of a large number of interrelated variables, PCA can be an effective method to reduce the dimensionality of the data set which just contains much fewer variables that are ordered so that the first few retain most of the variation present in all of the original variables.

Suppose we are given a training set 
$$X = (X_1, X_2, … , X_N)$$ with dimensionality $D$ and we want to reduce it to dimensionality $M$.

1. Conduct the normalization using standard score: $Z_i = \frac{X_i-\mu}{\sigma}$ where $i = 1, 2, … , N$, $\mu = \frac{\sum\limits_{i=1}^{N}X_i}{N},  \sigma^{2} = \frac{\sum\limits_{i=1}^{N}(X_i-\mu)^{2}}{N}$

2. Calculate the covariance matrix $S$ of $Z$: $S = \frac{\sum\limits_{i=1}^{N}(X_i-\mu)(X_i-\mu)^{T}}{N}$

3. Calculate the eigenvectors $u$ and eigenvalues $\lambda$ of $S$: Sort the eigenvectors $u_i$ according to their associated eigenvalue$\lambda_1\ge\lambda_2\ge...\ge\lambda_M$

4. Discard smaller eigenvalues, Each principal component $U_i$ can be written as: $Z_i = \sum\limits_{j=1}^{M}b_{ij}u_j$ where the scalars  $b_{ij}$ are the coordinates of $Z_i$ in the principal component space


#### 3.1.4 Feature learning and extraction by K means

Besides PCA, we can use K means method to do the feature learning and extraction$^5$.

**Feature learning**

First, we extract P unique patches from each image. So the total number of patches will be $P\times N$, N is the number of images. Each patch will have a shape of $(w, w, 3)$, then convert these matrices into vectors. Each vector’s length is  $w\times w\times 3$.  After extracting $P\times N$ patches, calculate the standard score for each vector, and then use the K means function to calculate K centroids for these vectors. K means clustering is an unsupervised learning method, which aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. After clustering $P\times N$ patches into K classes, we can get K centroids.

**Feature extraction**

For each image in the dataset, we perform convolutional extraction by extracting $(32-w+1)^2$ patches, each patch with a shape of (w, w, 3).  We convert each patch into a $w\times w\times 3$ vector, and use K centroids to predict this patches’ class. For every vector, we will get a K length vector, where the number on minimal distance index position is 1, and the others are 0. For each image, we will get a $(32-w+1)^2\times K$ values, which is extremely large.

So we need to use pooling method, for this project we conduct mean pooling instead of max pooling, because using max pooling will have a great chance to get a K length vector that every position number is 1. We divide each image into 4 quadrants, and for each quadrant, we calculate mean value of $(32-w+1)^2/4$ vectors. After calculating pooling for 4 quadrants, we can get 4 vectors with length K. then combine these 4 vectors together to get a vector with length $4\times K$, which is the new features for the dataset.

### 3.2 Models and Design choices

#### 3.2.1 KNN

KNN is short for k-nearest neighbours, which means we need to calculate the distances between 1 test data with all training dataset, and then find k-smallest distance. Count the number of different classes, and use the most frequently class the predicted class. This machine learning method quite time-consuming since comparing with other methods which has fit and prediction phase, this method only has predict phase, for every new test record, we need to calculate all the distances.

#### 3.2.2 Naive Bayes

Naive Bayes applies Bayes theorem with strong (naive) independence assumptions between the features to conduct probabilistic classification. 

Suppose each record $X$ in the data set has $n$ features $X={x_1, x_2, ... , x_n}$. And there are $m$ classes $C={C_1, C_2, ... , C_m}$. If the Naive Bayes classifier consider $X$ to be in class $C_i$, $P(C_i|X)>P(C_j|X)$ where $1\le{i}\le{m}$, $1\le{j}\le{m}$, and $j\ne{i}$.

According to Bayes theorem, because $P(X)$ is a constant for all classes, maximum posterior $P(C_i|X)$ is equivalent to maximum the joint probability $P(C_i,X)$ which can be rewritten as:
$P(C_i,X)=P(C_i,x_1,x_2,...,x_n)$

$=P(x_1|x_2,x_3,...,x_n,C_i)P(x_2|x_3,x_4,...,x_n|C_i)...P(x_{n-1}|x_n,C_i)P(x_n|C_i)P(C_i)$

Because of the independence assumptions, $P(x_k|x_{k+1},x_{k+2},...,x_n,C_i)=P(x_k|C_i)$ where $1\le{j}\le{m}$. 

Then $P(C_i,X)=P(C_i)P(x_1|C_i)P(x_2|C_i)...P(x_n|C_i)=P(C_i)\prod\limits_{k=1}^{n}P(x_k|C_i)$

Thus, the problem is equivalent to find i which can maximum $P(C_i)\prod\limits_{k=1}^{n}P(x_k|C_i)$

#### 3.2.3 Random Forest

Random Forest is a machine learning method on the foundation of decision tree classification. It creates multiple decision trees at training time. The forest output takes the majority vote in the case of classification trees. This method can efficiently get rid of overfitting to the training set when simply using decision tree classification.

<img src="https://i.imgur.com/5r52ttQ.png" width="50%" />
<center><b>Figure 2.1</b> Random Forest Model</center>


Random Forest uses bootstrap aggregating, which is also known as bagging, to tree learners. Suppose we are given a training data set $X = (x_1, x_2, … , x_N)$with labels $Y = (y_1, y_2, … , y_N)$. This method repeatedly, for $M$ times, selects a random subset with the replacement of the features at each candidate split in the learning process and fit trees, typically using CART (Classification And Regression Tree) Algorithm.

For $m = 1, 2, ... , M$:

1. Each $x_i$ ($i = 1, 2, ... , N$) has $K$ features which are $x_{i1}, x_{i2}, … , x_{iK}$

2. Sample, with replacement, $j$ ($j < K$) features from $x_i$; call these $X_m$.

3. Train a classification tree $f_m$ on $X_m$ and $y_i$ (the label of $x_i$)

After training, predictions for unseen feature $x_{il}$ can be made by taking the majority vote in the case of classification trees. By doing this, if some features are very strong predictors for the outputs, these features will be selected in many of the $M$ trees, causing them to become correlated. For random forest classification problem, typically, $\sqrt{K}$ (round down) features are used in each split.


#### 3.2.4 Multilayer Perceptron
A multilayer perceptron (MLP) is a supervised learning algorithm which based on the *feedforward artificial neural network*. Each node in each layer is fully connected to the next layer as the figure shows below (Figure 3.2).

<img src="https://i.imgur.com/K65rhDQ.png" width="30%">

<center><b>Figure 3.2</b> Model of Multilayer Perceptron</center>


Formally, a one-hidden-layer MLP is a function 
$$f:R^D \rightarrow R^L$$D is the input vector’s size and L is the size of output vector $f(x)$(total number of classes).Such that, in matrix notation:$f(x)=G(b^{(2)}+W^{(2)}(S(b^{(1)}+w^{(1)}x)))$with bias vectors $b^{(1)}$,$b^{(2)}$. weight matrices  $W^{(1)}$,$W^{(2)}$ and activation functions G and S.

The vector $h(x)=\phi(x)=S(b^{(1)}+w^{(1)}x)$ constitutes the hidden layer $W^{(1)}\in R^{D*D_h}$  is the weight matrix connecting the input vector to the hidden layer. Each column  $W_i^{(1)}$represents the weights from the input units to the $i-th$ hidden unit. Nodes in intermediate layer typical choices for s is the *logistic sigmoid function*, with $sigmoid(a)=\frac{1}{1+e^{-a}}$  $sigmod$ are scalar-to-scalar functions but their natural extension to vectors and tensors consists in applying them element-wise (e.g. separately on each element of the vector, yielding a same-size vector).

Nodes in output layer use Softmax function:  $f(z_i)=\frac{e^{z_i}}{\sum_{k=1}^Ne^{z_k}}$For this task,due to the total number of classes is 10,the minimum size of output size must be 10.


#### 3.2.5 CNN
CNN is short for Convolutional Neural Network, which is a class of deep, feed-forward artificial neural networks, most commonly applied to analyzing visual imagery.

**Convolutional layer**: This layer's parameters consist of a set of learnable filters, each filter is a is small spatially. During the forward pass, compute dot product between the entries of the filter and the input, and produce a 2-dimensional activation map of that filter. 

**Pooling Layer**: Its function is to progressively reduce the spatial size of the representation to reduce the number of parameters and computation in the network, and hence to also control overfitting. Usually, we use max operations to do pooling.

**Fully-connected layer**: This layer connects every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional MLP.



## 4. Experiments 
For all accuracy and execution time, if not especially proposed, 10-fold cross validation is used. The execution environment is colab with GPU, which is 2 Intel(R) Xeon(R) CPU @ 2.30GHz, RAM 13GB, GPU Tesla K80 if used. Execution time is the time of just doing the 10-fold time cross validation, it doesn't include preprocessing such as PCA. When talking about accuracy, it means the overall accuracy which is the number of right prediction divide by the total number of tests. When talking about accuracy for each label, it means $(TP+TN)/(TP+TN+FP+FN)$.

### 4.1 KNN with PCA
There are two parameters when doing KNN with PCA, the dimension of D for PCA reduction dimension and K for KNN's K-nearest neighbours. Table 4.1 shows the overall accuracy and execution time for different D and K combinations with a 10-fold cross validation. Because the execution time for KNN highly depends on the dimension of the dataset, so we should reduce the dimension to a not large number. From our former experience about using KNN method with MNIST, we found K=8 is the most suitable for MNIST. So we use K=8 to evaluate the D's influence on the dataset.

<center><b>Table 4.1</b> accuracy and execution time for KNN when K=8</center>

| D    | K    | accuracy | time(s) |
| ---- | ---- | -------- | ------- |
| 10   | 8    | 0.3501   | 27      |
| 20   | 8    | 0.4052   | 75      |
| 30   | 8    | 0.4158   | 100     |
| 40   | 8    | 0.4086   | 148     |
| 50   | 8    | 0.4032   | 192     |

From table 4.1, we can see the execution time increases as D increases, but the accuracy first increases and then decreases. If D is large, there might be noise still need to be reduced. If D is small, the images might lose some features. Considering the execution time and dimension, we consider D=30 is the best choice when doing KNN. Then we use D=30 and evaluate K's influence on the accuracy and execution time.

<center><b>Table 4.2</b> accuracy and execution time for KNN when D=30</center>

| D      | K      | accuracy   | time(s) |
| ------ | ------ | ---------- | ------- |
| 30     | 4      | 0.3976     | 93      |
| 30     | 6      | 0.4078     | 96      |
| 30     | 8      | 0.4158     | 100     |
| 30     | 10     | 0.4175     | 101     |
| 30     | 12     | 0.4199     | 104     |
| **30** | **14** | **0.4205** | **106** |
| 30     | 16     | 0.4195     | 107     |
| 30     | 18     | 0.4189     | 108     |


From table 4.2, we can see the execution time is similar, but the accuracy first increases and then decreases. If K is small, the result is sensitive to noise points. If K is large, neighbourhoods may include points from other classes. For this project, we finally choose D=30, K=14. Figure 4.1 shows the confusion matrix and table 4.3 shows the accuracy, precision, recall, and f1 score for each label.

<center><b>Table 4.3</b> KNN accuracy, precision, recall, and f1 score for each label</center>

| Label      | Accuracy | Precision | Recall | F1_Score |
| ---------- | -------- | --------- | ------ | -------- |
| airplane   | 0.8887   | 0.4554    | 0.5773 | 0.5092   |
| automobile | 0.9149   | 0.6119    | 0.4075 | 0.4892   |
| bird       | 0.8403   | 0.2879    | 0.4053 | 0.3367   |
| cat        | 0.8817   | 0.3515    | 0.2168 | 0.2682   |
| deer       | 0.8402   | 0.2959    | 0.4333 | 0.3517   |
| dog        | 0.8963   | 0.4671    | 0.2616 | 0.3354   |
| frog       | 0.8543   | 0.3529    | 0.5477 | 0.4292   |
| horse      | 0.9120   | 0.5967    | 0.3707 | 0.4573   |
| ship       | 0.8998   | 0.4993    | 0.6378 | 0.5601   |
| truck      | 0.9131   | 0.6159    | 0.3488 | 0.4454   |

<img src="https://i.imgur.com/pSS7EW3.png" width="50%">

<center><b>Fig 4.1</b> Confusion Matrix for KNN</center>

### 4.2 Naive Bayes with PCA and K means
Because after PCA, we will get some negative values, to make it easier to calculate, we use Gaussian Naive Bayes. There is only one parameter for Gaussian Naive Bayes with PCA, the dimension D. Table 4.4 shows the accuracy and execution time for different D. The accuracy reaches the top when D=30, which is 0.37.

<center><b>Table 4.4</b> ccuracy and execution time for Gaussian Naive Bayes</center>

| D      | accuracy   | time(s) |
| ------ | ---------- | ------- |
| 10     | 0.3306     | 0.8     |
| 20     | 0.3602     | 1.4     |
| **30** | **0.3705** | **1.4** |
| 40     | 0.3657     | 1.2     |
| 50     | 0.3540     | 1.4     |
| 60     | 0.3552     | 1.6     |
| 80     | 0.3460     | 1.7     |
| 100    | 0.3353     | 2       |


With feature learning and extraction, each dataset is a 4×K long vector, and the number is the mean pooling of the patches. So the number has a meaning here, the bigger it is, the closer the patch is to a certain feature. So we use  Multinomial Naive Bayes.

There are several parameters for k means feature learning and extraction. P is the number of patches extracted from each image when doing feature learning, w is the shape of each patch, K is the number of classes when doing K means. For K means fitting process, the total number of training patches is N×P, which is large. So we do partial_fit every 100 images to save time and memory. For prediction, we use the same method to save memory. Table 4.5 shows these parameters' influence on the execution time (including k means time)and accuracy.

<center><b>Table 4.5</b> Different P, w, K for K means</center> 

| P      | w     | K        | time     | accuracy   |
| ------ | ----- | -------- | -------- | ---------- |
| 50     | 7     | 400      | 543      | 0.4202     |
| 10     | 7     | 400      | 532      | 0.4221     |
| 10     | 7     | 800      | 850      | 0.4381     |
| **20** | **7** | **1200** | **1290** | **0.4481** |
| 20     | 9     | 800      | 628      | 0.4148     |


From first 2 rows, P doesn't influence the accuracy a lot since we have already got 60,000 records, the total number of patches is enough, and a larger number of patches might result in overfitting. From row 1,2,4, we can see that the increase of K will result in the increase of accuracy, but the execution time increases. The reason we use P=20 for K=1200 is that for every 100 images, we need more than 1200 patches to do partial_fit. And due to the limitation of colab's memory, we can't apply a bigger K for this project.  From row 3,5, we can see that w=7 is more suitable. For this project, we finally choose P=20, w=7, K=1200. Figure 4.2 shows the confusion matrix and table 4.3 shows the accuracy, precision, recall, and f1 score for each label.

<center><b>Table 4.6</b>  NB accuracy, precision, recall, and f1 score for each label</center>

| Label      | Accuracy | Precision | Recall | F1_Score |
| ---------- | -------- | --------- | ------ | -------- |
| airplane   | 0.8967   | 0.4832    | 0.4758 | 0.4795   |
| automobile | 0.9146   | 0.5742    | 0.5646 | 0.5694   |
| bird       | 0.8869   | 0.3350    | 0.1325 | 0.1898   |
| cat        | 0.8727   | 0.3437    | 0.2995 | 0.3200   |
| deer       | 0.8759   | 0.3778    | 0.3723 | 0.3750   |
| dog        | 0.8804   | 0.4082    | 0.4358 | 0.4216   |
| frog       | 0.8608   | 0.3729    | 0.5741 | 0.4521   |
| horse      | 0.9091   | 0.5566    | 0.4468 | 0.4957   |
| ship       | 0.9007   | 0.5030    | 0.6133 | 0.5527   |
| truck      | 0.8981   | 0.4918    | 0.5661 | 0.5264   |

<img src="https://i.imgur.com/V5udnoA.png" width="50%">

<center><b>Fig 4.2</b> Confusion Matrix for K means and Naive Bayes</center>

For Naive Bayes, we increase the overall accuracy from 37% to 44.8% by using k means feature extraction and learning instead of simply using PCA.

### 4.3 Random Forest with PCA
The execution environment of Random Forest is 1 Intel(R) Core i7 CPU @ 2.50GHz, RAM 16 GB. There is no GPU used in this step.

The execution time for Random Forest highly depends on the dimension of the features of each candidate because the number of features to consider for split is directly related to the total feature number of each candidate. We should reduce the dimension to a smaller number.

We focused on adjusting three main parameters when doing Random Forest with PCA. These parameters are the number of principal components for PCA reduction dimension, the number of decision trees constructed in the forest, and max_features for the number of features to consider when looking for the best split. 

Figure 4.2 shows the overall accuracy of the combinations of different tree number and max_features. For the tree number, we chose 50 to 1000, with step of 50. And three max_features, "sqrt", "log2", and "none", are used to split the features. "Sqrt" means the number of features we used for split is the square root of the total number of the features of each candidate. "Log2" means we random select $log_2N$ (N equals to the total number of the features of each candidate) features to fit one tree. If the max_feature is "none", we use all the features. The number of principal components is set to be 130 when conducting PCA.

<img src="https://i.imgur.com/G3zLJOx.png" width="70%">

<center><b>Figure 4.3</b> Accuracy for random forest</center>

From figure 4.3, the "sqrt" is the best max_feature and the accuracy increases with the number of trees in the forest. Take the execution time without a 10-fold cross validation, which is shown in figure 4.3, into consideration, we consider the "sqrt" to be the best max_feature and 550 to be the best tree number when using random forest learn this data set.

<img src="https://i.imgur.com/w6iN21A.png" width="50%">

<center><b>Figure 4.4</b> The execution time of different tree number</center>

From figure 4.5, we can see the influence of the accuracy by the different tree number and the different principal component (PC) number of PCA. It is obvious that 80 is the best PC number for this procedure.

<img src="https://i.imgur.com/MSgSgLZ.png" width="70%">

<center><b>Figure 4.5</b> Accuracy based on Tree Number and PC Number</center>

Then we run random forest with a 10-fold cross validation while setting **"sqrt"** to be the max_feature, **550** to be the tree number, and **80** to be the PC number. The average accuracy is **0.5014**. And the execution time is 2958 seconds, as well as 49 minutes 18 seconds. Figure 4.6 shows the confusion matrix and table 4.4 shows the accuracy, precision, recall, and f1 score for each label.

<img src="https://i.imgur.com/pWpeDka.png" width="45%">

<center><b>Fig 4.6</b> Confusion Matrix for K means and Naive Bayes</center>

<center><b>Table 4.7</b> Random Forest accuracy, precision, recall, and f1 score for each label</center>

| Label      | Accuracy | Precision | Recall | F1_Score |
| ---------- | -------- | --------- | ------ | -------- |
| airplane   | 0.9147   | 0.5752    | 0.5626 | 0.5688   |
| automobile | 0.9125   | 0.5553    | 0.6288 | 0.5898   |
| bird       | 0.8914   | 0.4421    | 0.3278 | 0.3764   |
| cat        | 0.87953  | 0.3759    | 0.3095 | 0.3395   |
| deer       | 0.8944   | 0.4694    | 0.4240 | 0.4455   |
| dog        | 0.8864   | 0.4281    | 0.4041 | 0.4158   |
| frog       | 0.8971   | 0.4881    | 0.5998 | 0.5382   |
| horse      | 0.9134   | 0.5750    | 0.5155 | 0.5436   |
| ship       | 0.9160   | 0.5691    | 0.6590 | 0.6107   |
| truck      | 0.8955   | 0.4814    | 0.5751 | 0.5241   |


### 4.4 Multilayer Perceptron with PCA 
Before doing experiments, we suppose a hypothesis that the more hidden layers and neurons, the higher accuracy as well. Thus, we have tested different sizes of hidden layers and nodes in the following part. Firstly, in order to train the model and raise the accuracy, we reduce the dimension of data from 3072 to 200 and randomly set only one hidden layer with the size of 150 without 10-fold.


<center><b>Table 4.8</b> Execution time and accuracy with or without PCA</center>

| Dimension | time    | Accuracy |
| --------- | ------- | -------- |
| 3072      | 519.23s | 48.8%    |
| 200       | 86.056s | 48.41%   |

As the results shown in Table 4.8, the execution time is much longer without dimension reduction but the accuracy is similar. So it is good for us to do PCA before doing MLP. We set 200 as the fixed value for dimension reduction and compare the result of various numbers of hidden layers.               

<center><b>Table 4.9</b> Accuracy and Execution time when D=200</center>

| hidden layer | accuracy | time(s) |
| ------------ | -------- | ------- |
| (100,)       | 0.5042   | 90      |
| (100,50)     | 0.4585   | 142     |
| (100,50,10)  | 0.4551   | 181     |

The result of experiments as shown in Table 4.9. With the increasing number of hidden layers, the accuracy decreases a little but execution time increases. So we decide to use only 1 hidden layer for this project, which can save time and increase accuracy.  Then we use different combinations for PCA's D and the size of the only one hidden layer K.







<center><b>Table 4.10</b> Accuracy and Execution time for different K,D combination</center>

| D       | Hidden Layer | accuracy   | time(s) |
| ------- | ------------ | ---------- | ------- |
| 100     | (50,)        | 0.5245     | 52      |
| 100     | (100,)       | 0.5310     | 58      |
| **100** | **(150,)**   | **0.5328** | **62**  |
| 100     | (200,)       | 0.5138     | 72      |
| 200     | (50,)        | 0.5136     | 72      |
| 200     | (100,)       | 0.5142     | 90      |
| 200     | (150,)       | 0.4932     | 143     |
| 50      | (50,)        | 0.5116     | 59      |
| 50      | (100,)       | 0.5100     | 36      |


For one hidden layer, the accuracy will decrease when the size is large. The reason might be that when the size is too large, there is a risk of overfitting. For this project we choose **PCA D=100, hidden layer number=1, layer size=150** to get the final result for Multilayer Perceptron.  The overall accuracy of conducting 10-fold cross validation is **0.5210**. 

<center><b>Table 4.11</b> MLP accuracy, precision, recall, and f1 score for each label</center>

| Label      | Accuracy | Precision | Recall | F1_Score |
| ---------- | -------- | --------- | ------ | -------- |
| airplane   | 0.9131   | 0.5633    | 0.5848 | 0.5738   |
| automobile | 0.9250   | 0.6251    | 0.6251 | 0.6251   |
| bird       | 0.8820   | 0.4056    | 0.3868 | 0.3960   |
| cat        | 0.8757   | 0.3658    | 0.3308 | 0.3474   |
| deer       | 0.8909   | 0.4538    | 0.4475 | 0.4506   |
| dog        | 0.8863   | 0.4288    | 0.4106 | 0.4195   |
| frog       | 0.9125   | 0.5575    | 0.6068 | 0.5811   |
| horse      | 0.9136   | 0.5635    | 0.6055 | 0.5837   |
| ship       | 0.9295   | 0.6443    | 0.6601 | 0.6521   |
| truck      | 0.9129   | 0.5663    | 0.5511 | 0.5586   |

<img src="https://i.imgur.com/HOWai62.png" width="45%">

<center><b>Fig 4.7</b> Confusion Matrix for Multilayer Perceptron</center>


### 4.5 CNN

Typically, CNN structure follows the pattern:

```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
```

For this project, we use keras to calculate CNN. To determine the parameters we need to choose, we do modelling on 50,000 training dataset and 10,000 test set. After determining the parameters, we use 10-fold cross validation to get confusion matrix and precision, recall, F1_Score. Table 4.12 shows the accuracy and execution time for different parameters when epoch=50.

| N     | M     | K     | accuracy   | time(s)    |
| ----- | ----- | ----- | ---------- | ---------- |
| 1     | 2     | 1     | 0.7665     | 455.75     |
| **2** | **2** | **1** | **0.8100** | **770.50** |
| 2     | 1     | 1     | 0.7327     | 1004.10    |
| 1     | 2     | 2     | 0.7666     | 478.19     |

<center><b>Table 4.12</b> Accuracy and execution time for epoch=50</center>

For these 4 combinations of parameters, we can get some ideas about N, M, K's influence on the accuracy. N=2 is better than N=1, M=2 is better than M=1, K=2 is better than K=1. But among all three parameters, N's influence is the biggest. For N=2, M=2, K=1, Figure 4.8 shows the accuracy of the training set and testing set. We see that when epoch is larger than 20, the accuracy will not increase a lot. Due to the limitation of GPU capability, we only use epochs=20 for 10-fold cross-validation.

<img src="https://i.imgur.com/hHnSjHz.png" width="50%">

<center><b>Figure 4.8</b> Model Accuracy for different Epoch values</center>

Figure 4.9 shows the confusion matrix and Table 4.13 shows the accuracy, precision, recall, and f1 score for each label for [[CONV -> RELU]*2 -> POOL]*2 -> [FC -> RELU]*1. The average accuracy is **0.8093**, the execution time is **3311s**.









<center><b>Table 4.13</b> Accuracy and execution time for epoch=50</center>

| Label      | Accuracy | Precision | Recall | F1_Score |
| ---------- | -------- | --------- | ------ | -------- |
| airplane   | 0.9665   | 0.8338    | 0.8315 | 0.8326   |
| automobile | 0.9817   | 0.9123    | 0.9043 | 0.9083   |
| bird       | 0.9465   | 0.7338    | 0.7308 | 0.7323   |
| cat        | 0.9287   | 0.6426    | 0.6480 | 0.6453   |
| deer       | 0.9553   | 0.7659    | 0.7968 | 0.7811   |
| dog        | 0.9459   | 0.7435    | 0.7008 | 0.7215   |
| frog       | 0.9689   | 0.8235    | 0.8773 | 0.8495   |
| horse      | 0.9693   | 0.8655    | 0.8206 | 0.8425   |
| ship       | 0.9769   | 0.8775    | 0.8943 | 0.8858   |
| truck      | 0.9767   | 0.8865    | 0.8795 | 0.8830   |

<img src="https://i.imgur.com/9T57XCR.png" width="50%">

<center><b>Figure 4.9</b> Confusion Matrix for CNN</center>




## 5. Conclusion
### 5.1 Comparison between different methods
Table 5.1 shows the overall accuracy and execution time for different methods.

KNN with PCA is fast but has the lowest accuracy for this project, besides, when the number of records increases, the time will increase a lot. So we don't think it is a good method for this project. Naive Bayes with K means feature learning and extraction is faster than other methods and has similar accuracy as KNN. The disadvantage for this method is similar to KNN since it needs to predict K means classes for each patch, which will cost a lot of time. Random Forest and Multilayer Perceptron with PCA are not bad. Both methods can get the accuracy about 50%. The execution time is not so sensitive to the size of the test set. When equipped with GPU, CNN is a very good method for its high accuracy and fast execution time. 

CNN is better for this project or saying, better on images processing, it takes advantage of inherent properties of images. Some features extracted are meaningless for human, but CNN model can make use of them.

<center><b>Table 5.1</b> Comparison with different Method</center>

| Preporcessing | Method                | Accuracy | Time(s) | CPU                            | GPU(if used) |
| ------------- | --------------------- | -------- | ------- | ------------------------------ | ------------ |
| PCA           | KNN                   | 0.4205   | 106     | Intel(R) Xeon(R) CPU @ 2.30GHz |              |
| K means       | Naive Bayes           | 0.4481   | 1200    | Intel(R) Xeon(R) CPU @ 2.30GHz |              |
| PCA           | Random Forest         | 0.5014   | 2958    | Intel(R) Core i7 CPU @ 2.50GHz |              |
| PCA           | Multilayer Perceptron | 0.5210   | 672     | Intel(R) Xeon(R) CPU @ 2.30GHz |              |
| /             | CNN                   | 0.8093   | 3311    | Intel(R) Xeon(R) CPU @ 2.30GHz | Telsa K80    |

### 5.2 Confusion Matrix and F1 Score analysis for different labels

For all 5 methods, the F1 Scores for cat, dog, bird and deer are clearly lower than other labels, which means it is difficult for these methods to classify these images. That is because they are all not big animals, and they are more similar with each other. KNN method can show the similarity among each labels, from KNN's confusion matrix, we can see that these 4 animals are often confused with each other, which means they are more similar with each other than other 6 labels.

For K mean + Naive Bayes, the label bird's f1 score is the smallest, by randomly printing out birds images, we find that even ostriches and chickens are considered as birds, which makes it harder for k means to get the features from "bird" label.

From the CNN Confusion Matrix, we can see that these label 3 and label 5 are often miss classified with each other. These 2 labels represent cat and dog, the reason for this situation might be that when CNN learning the features, it can get the shape of the target, the fur texture of the target, but cats have the similar shape and fur texture with dogs. Actually, it is difficult for human to distinguish these 2 labels by just looking at 32*32 images. For labels like plane and ship, the f1 scores are high, CNN does a good job on these labels, the reason should be that the features of the plane and ship are more unique.


### 5.3 Future Work

For K means feature learning and extraction, besides our hard K means method, there is a "soft" K means method$^3$. When predicting the classes for patches, use $f_k(x)=max\{0, \mu(z)-z_k\}$ , where  $z_k = ||x−c(k)||^2$ and $µ(z)$ is the mean of the elements of $z$. Due to the limitation of colab's memory space, we didn't do this for this project. We want to try this method of feature extraction and to see whether it will improve the accuracy for Multinomial Naive Bayes.

For CNN method, we just tried 4 convolutional layers and 2 max pooling layers in total, we want to try other combination of layers to see how these layers influence the performance and how the parameters in each layer influence the performance.

There are some labels' f1 scores smaller than others'. We want to do some research on how to deal with imbalanced classes and improve the accuracy of those labels.

For this project, we only use a single model for each method, we also want to try ensemble learning by combining different models together to see whether it can improve the accuracy.




## References
1. L. Wan, M. Zeiler, S. Zhang, Y. L. Cun, and R. Fergus, Regularization of neural networks using dropconnect, in Proceedings of the 30th International Conference on Machine Learning (ICML-13), 2013, pp. 1058–1066
2. Benjamin Graham, Fractional Max-Pooling, 2014
3. Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller, Striving For Simplicity:
    The All Convolutional Net, ICLR, 2015
4. Jasper Snoek, Hugo Larochelle, and Ryan P. Adams, Practical Bayesian Optimization of Machine Learning Algorithms, NIPS, 2012
5. A. Coates, A. Y. Ng, and H. Lee. An analysis of single-layer networks in unsupervised feature learning. In International Conference on Artificial Intelligence and Statistics, pages 215–223, 2011
