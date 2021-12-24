---
author: Andrae Lerone Lewis
title: Music Curation with Neural Networks
subtitle: "Professor: Ivan Ivanov"
abstract: |
  This experiment aims to replicate a Spotify-like recommendation algorithm. Using a Neural Network trained on data obtained from the Spotify Web API, a model was created using Keras that could predict a user's preferences based on the musical characteristics of a track with an accuracy of 70%.


nocite: |
  @*
---

# Introduction

This experiment aims to create a model that could curate music for a user. The model, a Neural Network, will take the musical characteristics of a track, obtained from Spotify's Web API, as inputs and output whether it predicts that the user will like or dislike the track.

## Neural Networks: A Basic Overview

A Neural Network is a collection of neuron organized into layers. Data is input into the the Input Layer, processed by Hidden Layers, and a result comes out of the Output Layer. Each neuron's value is calculated as follows:
$$\sum_{i=1}^n w_i\times x_i + bias$$
where $n$ is the number of neurons from the previous layer, $x_i$ is the $i$th neuron from the previous layer, $w_i$ is the weight of the connection between said neuron and the current neuron, and $bias$ "an additional parameter used along with the sum of the product of weights and inputs to produce an output." [@Nduati2020] The result of this calculation is passed through an activation function, which determines whether or not it is passed on to the next neuron. This calculation is done for every neuron in a layer, for every layer in the network. Once it reaches the output layer, the value of the output neuron is compared to the actual value. If the output of the network is incorrect, backpropagation occurs: the weights are readjusted to minimize error. [@Nduati2020]

The neural network used in this experiment is created using the `tf.keras.Sequential` class, a simple network created by adding layers together. The kind of layers created come from the class `tf.keras.layers.Dense`, which behave according to the formula described above. Dropout layers (from `tf.keras.layers.Dropout`) are placed at two points in the network to reduce the risk of overfitting. They function by setting some of its inputs to 0 and rescaling those unaffected in such a way that the sum is unchanged. [@KerasDropout]

The activation functions used in the model function as follows:

`relu`: $\max(x,0)$

`softplus`: $\log(\exp(x) + 1)$

`softmax`: The softmax of each vector x is computed as $\exp(x) / tf.reduce_sum(\exp(x))$ [@KerasActivation]

`sigmoid`: $1/(1+\exp(-x))$

<!-- ## A.I. in Content Curation -->




# Methodology

The neural network is trained using data found in [this](https://www.kaggle.com/geomack/spotifyclassification) Kaggle dataset.[@McIntire] The data is someone's spotify preferences as well as the information Spotify has on each song. The Neural Network is trained on this data, with the goal of predicting whether or not a song will be liked based on the Spotify's analysis. 

## Spotify's Song Analysis

Spotify uses a variety of metrics to categorize music on its platform. This data is accessible through the [Spotify Web API](https://developer.spotify.com/documentation/web-api/). The song properties in the dataset come from the API endpoint `/audio-features/`. Some are self explanatory: `accousticness` is whether or not a song is accoustic, `instrumentalness` a measure of how instrumental a song is[@Spotify2021Dev], and `liveness` how confident they are that the song was performed live. `Danceability` is generally the measure of musical regularity, and `speechiness` is how much it ressembles speech, rap and spoken word being a $1$.
`valence` is simply a measure of how "happy" a song is, or how positive the message is, with $1$ being happy or positive.  Other information (`key`, `duration_ms`, `mode`, `song_title`, `artist`, `Unnamed: 0`, `tempo`, `loudness`, `time_signature`) from the endpoint is removed from the dataset before training. `song_title` and `artist` were removed to avoid artist bias, that is one artist being over-represented in the dataset and having a higher weight than musical features. `energy` is a little strange. 
"Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy."[@Spotify2021Dev]
\pagebreak

## The Code

### Setup

First, the code imports all the packages and libraries needed.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Average, BatchNormalization
from keras.callbacks import EarlyStopping
```

Next the data from the Kaggle dataset is loaded in.

```python
df = pd.read_csv('data.csv')
```

Next, the data being disregarded is dropped from the dataset.

```python
# variable filtering
df.drop(['key'], axis=1, inplace=True)
df.drop(['duration_ms'], axis=1, inplace=True)
df.drop(['mode'], axis=1, inplace=True)
df.drop(['song_title'], axis=1, inplace=True)
df.drop(['artist'], axis=1, inplace=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['tempo'], axis=1, inplace=True)
```

The data set is split into our inputs (`X`) and our output (`Y`).

```python
# split into X and Y
Y = df['target']
X = df.drop(['target'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
```

### The model

The model is then initiated and built. The parameters were figured out using trial and error. The size and types of layers were first chosen at random then manually adjusted, with changes that increased accuracy kept and those lowering it discarded. Accuracy assessment is elaborated on further down. This model could be vastly improved with more time to tweak it and more sophisticated tools.

```python
model = Sequential()
model.add(Dense(10, input_shape=(X.shape[1],), activation='relu')) 
model.add(Dense(10, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softplus'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softplus'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dropout(0.1))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

Now the model is compiled. Again, parameters were determined by trial and error. Given more time and resources, these too could be further optimized. 
The `Adam` model is described as "computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters." [@kingma2017adam]
They also say that the parameters require "little tuning" which is perfect for this experiment. The two metrics, `accuracy` and `BinaryAccuracy`, are used within our loss function `binary_crossentropy`. 
These metrics are what we seek to maximize while minimizing the loss function. 
`EarlyStopping` is used to prevent the model from training endlessly while very minute changes are occuring within the `accuracy` and `BinaryAccuracy` metrics. 
This prevents the waste of computational resources and iterations on the algorithm to occur faster. 
The patience of `50` means that the training stops after 50 epochs of BinaryAccuracy not improving. 
The `batch_size` and `validation_split` are simply set to values that worked best without taking a lot of time to compute.
A larger batch size didn't improve accuracy and made the model training take significantly longer.
\clearpage

```python
model.compile(
	optimizer='Adam', 
	loss='binary_crossentropy',
	metrics=['accuracy','BinaryAccuracy']
) 
es = EarlyStopping(
	monitor='val_binary_accuracy', 
	mode='max', # don't minimize the accuracy!
	patience=50,
	restore_best_weights=True
)
history = model.fit(
	X_train,
	Y_train,
	callbacks=[es],
	epochs=500,
	batch_size=75,
	validation_split=0.2,
	shuffle=True,
)
```

This part of the code shows an analysis of the model's training. The data here will be explored further in the [training part of the results section](#training). 

```python
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss'] 
val_accuracy = history_dict['val_binary_accuracy'] 

epochs = range(1, len(loss_values) + 1) 

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.plot(epochs, val_accuracy, 'red', label='Validation Accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

This part of code is for testing the accuracy of the model. 
First, some packages are imported, and a `test` function is defined. 
This function takes an `X` and `Y`, which are the inputs and the expected outputs respectively. 
The model is then used to make predictions from the inputs `X`, into a new variable `preds`. 
These predictions are rounded to create clean classifications, as the `predict` function returns a float. Using the predictions `preds` and the actual values `Y`, a confusion matrix is generated. A confusion matrix shows the types errors made by the algorithm.
Following the confusion matrix, a full classification report is generated. 

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def test(X, Y):
	preds = np.round(model.predict(X),0)
	print(confusion_matrix(Y, preds))
	print(classification_report(Y, preds))
```

Tests are then conducted on the testing data, the training data, and the whole dataset.

```python
test(X_test, Y_test)
test(X_train, Y_train)
test(X,Y)
```

# Results

## Training

<!-- ![Loss graph](./assets/loss.png) -->
\begin{figure}[h]
\caption[Training Loss and Accuracy Graph]{Graph of Training loss, Validation Loss, and Validation Accuracy per Epoch}
\centering
\subfile{assets/loss.pgf}
\end{figure}

\begin{figure}[h]
\caption[Training Loss and Accuracy Graph]{Graph of Training loss, Validation Loss, and Validation Accuracy per Epoch}
\centering
\subfile{assets/loss_overfitted.pgf}
\end{figure}

\clearpage

## Testing Results

### Under 500 Epochs

\begin{figure}[H]
\caption[Dataset Confusion Matrix]{Confusion Matrix for Predictions on the entire Dataset}
\centering
\subfile{assets/overall_data.pgf}
\end{figure}
\subfile{tables/overall_data}

\begin{figure}[H]
\caption[Training Confusion Matrix]{Confusion Matrix for Predictions on the Training Data}
\centering
\subfile{assets/training_data.pgf}
\end{figure}
\subfile{tables/training_data}

\begin{figure}[H]
\caption[Test Data Confusion Matrix]{Confusion Matrix for Predictions on the Test Data}
\centering
\subfile{assets/test_data.pgf}
\end{figure}
\subfile{tables/test_data}

### Over 1000 Epochs

\begin{figure}[H]
\caption[1000 Epoch Dataset Confusion Matrix]{Confusion Matrix for the entire Dataset After Over 1000 Epochs}
\centering
\subfile{assets/overall_data_overfitted.pgf}
\end{figure}
\subfile{tables/overall_data_overfitted}

\begin{figure}[H]
\caption[1000 Epoch Test Data Confusion Matrix]{Confusion Matrix for the Test Data After Over 1000 Epochs}
\centering
\subfile{assets/test_data_overfitted.pgf}
\end{figure}

\subfile{tables/test_data_overfitted}

\begin{figure}[H]
\caption[1000 Epoch Training Confusion Matrix]{Confusion Matrix for the Training Data After Over 1000 Epochs}
\centering
\subfile{assets/training_data_overfitted.pgf}
\end{figure}
\subfile{tables/training_data_overfitted}


# Discussion and Analysis

## The model

The model's accuracy over all the testing configurations never exceeds 72%. 
This accuracy score isn't particularly impressive and could potentially be improved with by tweaking the model. 
Despite this low accuracy, this model could still be deployed and would function adequately. 
Looking at the results, the rate at which the model predicts the user will like a song when they actually wouldn't is ~9%, which is half as frequently as the rate at which it it predicts the user would dislike a song when they would actually like it (~19%).
This is a good thing from a curation perspective. It's a better that most of the time, the user isn't suggested tracks that they won't like. 

These results line up with the experiment presents by the dataset author.
Their model was able to get to an accuracy of about 75%. [@McIntire2017]


## How it could improved

This system could be improved with a much larger model that takes into account more factors. 
Taking artists into account, for example, might help improve accuracy, but that was outside of the scope of this project. 
Furthermore, having more user data to corroborate with. 
If user A's preferences are similar to user B's, one could use user A *and* user B's data to predict whether or not user B will like a song. 
Agglomerating user data would also help cover any missing data in a single user. 
More data would also help. This algorithm was only trained on 2017 pieces of music, while Spotify reports having "over 70 million tracks." [@Spotify2021News]
The dataset size being so small was a practical problem, acquiring data on users' listening preferences is difficult without being Spotify itself; the data in the dataset was manually collected by the creator. 
Manual data collection makes this process very time consuming and increases the difficulty of scaling. 
Further training on the current data won't improve the model. The consequences of further training on the current data (> 1000 epochs) varies from marginal improvements to overfitting, where the improvements are substantial but only occur when predicting using the training data. The model performs about the same on testing data when trained for 500 epochs and more than 1000 epochs. The accuracy of the model goes from 0.68 with under 500 epochs to 0.71 with close to 2000 epochs, a 3% increase with 4$\times$ the computation time.

## What Spotify Actually Does

Spotify not only benefits from a much larger dataset (over 70 million tracks) and many more users, they also benefit from more varied data.
Spotify tracks your listening history, when you skip songs, how you organize your playlist, as well as where you are when you listen to a track. [@Balaganur2020] Spotify's massive user base allows them to group people with similar listening habits together. [@Irvine2019] They also deploy varied algorithms that analyze not only user and the music, but playlists themselves (using Natural Language Processing). [@Irvine2019]


# Conclusions

This experiment demonstrates that it is possible to recommend music using a Neural Network. While the accuracy of the model leaves a little to be desired, this proof of concept works despite only having a relatively small dataset that only includes information about the music itself. Major music streaming services have much larger and more varied datasets to train their models on using both Neural Network and more sophisticated techniques such as Natural Language Processing showing that this approach is not only viable, but profitable.

# Bibliography

<div id="refs"></div>

# Appendix

## Source Code

Full repository found [here](https://github.com/shadowlerone/ESP).

\clearpage

## Model Summary

\subfile{tables/summary}

\clearpage

## Full Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Average, BatchNormalization
from keras.callbacks import EarlyStopping

df = pd.read_csv('data.csv')

df.info()

# variable filtering
df.drop(['key'], axis=1, inplace=True)
df.drop(['duration_ms'], axis=1, inplace=True)
df.drop(['mode'], axis=1, inplace=True)
df.drop(['song_title'], axis=1, inplace=True)
df.drop(['artist'], axis=1, inplace=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['tempo'], axis=1, inplace=True)

df.info()

# df['Unnamed: 0']

# split into X and Y
Y = df['target']
X = df.drop(['target'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# print(X.shape)
# print(Y.shape)

# convert to numpy arrays
# X = np.array(X)

model = Sequential()
model.add(Dense(10, input_shape=(X.shape[1],), activation='relu')) 
# Add an input shape! (features,)
model.add(Dense(10, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.1))
# model.add(Average(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softplus'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softplus'))
model.add(Dense(10, activation='relu'))
# for i in range(100):
#   model.add(Dense(5, activation="softplus"))
model.add(Dense(10, activation='softmax'))

model.add(Dropout(0.1))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
	optimizer='Adam', 
	loss='binary_crossentropy',
	metrics=['accuracy','BinaryAccuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = EarlyStopping(
	monitor='val_binary_accuracy', 
	mode='max', # don't minimize the accuracy!
	patience=50,
	restore_best_weights=True)

# now we just update our model fit call
history = model.fit(
	X_train,
	Y_train,
	callbacks=[es],
	epochs=500, # you can set this to a big number!
	batch_size=75,
	validation_split=0.2,
	shuffle=True,
	# verbose=1
)

history_dict = history.history
# Learning curve(Loss)
# let's see the training and validation loss by epoch

# loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
val_accuracy = history_dict['val_binary_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(loss_values) + 1) 

# plot
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.plot(epochs, val_accuracy, 'red', label='Validation Accuracy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def test(X, Y):
	preds = np.round(model.predict(X),0)
	# confusion matrix
	print(confusion_matrix(Y, preds)) # order matters! (actual, predicted)
	## array([[490,  59],   ([[TN, FP],
	##       [105, 235]])     [Fn, TP]])
	print(classification_report(Y, preds))

test(X_test, Y_test)

test(X_train, Y_train)

test(X,Y)
```
