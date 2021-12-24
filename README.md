---
author: Andrae Lerone Lewis
title: Music Curation with Neural Networks
subtitle: "Professor: Ivan Ivanov"
abstract: |
  This experiment aims to replicate a Spotify-like recommendation algorithm. Using a Neural Network trained on data obtained from the Spotify Web API, a model was created using Keras that could predict a user's preferences based on the musical characteristics of a track with an accuracy of 70%.

---

<div id="introduction">

# Introduction

</div>

This experiment aims to create a model that could curate music for a
user. The model, a Neural Network, will take the musical characteristics
of a track, obtained from Spotify’s Web API, as inputs and output
whether it predicts that the user will like or dislike the track.

## Neural Networks: A Basic Overview

A Neural Network is a collection of neuron organized into layers. Data
is input into the the Input Layer, processed by Hidden Layers, and a
result comes out of the Output Layer. Each neuron’s value is calculated
as follows:
$$\\sum\_{i=1}^n w_i\\times x_i + bias$$
where *n* is the number of neurons from the previous layer,
*x*<sub>*i*</sub> is the *i*th neuron from the previous layer,
*w*<sub>*i*</sub> is the weight of the connection between said neuron
and the current neuron, and *b**i**a**s* “an additional parameter used
along with the sum of the product of weights and inputs to produce an
output.”[1] The result of this calculation is passed through an
activation function, which determines whether or not it is passed on to
the next neuron. This calculation is done for every neuron in a layer,
for every layer in the network. Once it reaches the output layer, the
value of the output neuron is compared to the actual value. If the
output of the network is incorrect, backpropagation occurs: the weights
are readjusted to minimize error.[2]

The neural network used in this experiment is created using the
`tf.keras.Sequential` class, a simple network created by adding layers
together. The kind of layers created come from the class
`tf.keras.layers.Dense`, which behave according to the formula described
above. Dropout layers (from `tf.keras.layers.Dropout`) are placed at two
points in the network to reduce the risk of overfitting. They function
by setting some of its inputs to 0 and rescaling those unaffected in
such a way that the sum is unchanged.[3]

The activation functions used in the model function as follows:

`relu`: max (*x*,0)

`softplus`: log (exp(*x*)+1)

`softmax`: The softmax of each vector x is computed as
exp (*x*)/*t**f*.*r**e**d**u**c**e*<sub>*s*</sub>*u**m*(exp(*x*))[4]

`sigmoid`: 1/(1+exp(−*x*))

<div id="methodology">

# Methodology

</div>

The neural network is trained using data found in
[this](https://www.kaggle.com/geomack/spotifyclassification) Kaggle
dataset.[5] The data is someone’s spotify preferences as well as the
information Spotify has on each song. The Neural Network is trained on
this data, with the goal of predicting whether or not a song will be
liked based on the Spotify’s analysis.

## Spotify’s Song Analysis

Spotify uses a variety of metrics to categorize music on its platform.
This data is accessible through the [Spotify Web
API](https://developer.spotify.com/documentation/web-api/). The song
properties in the dataset come from the API endpoint `/audio-features/`.
Some are self explanatory: `accousticness` is whether or not a song is
accoustic, `instrumentalness` a measure of how instrumental a song
is,[6] and `liveness` how confident they are that the song was performed
live. `Danceability` is generally the measure of musical regularity, and
`speechiness` is how much it ressembles speech, rap and spoken word
being a 1. `valence` is simply a measure of how “happy” a song is, or
how positive the message is, with 1 being happy or positive. Other
information (`key`, `duration_ms`, `mode`, `song_title`, `artist`,
`Unnamed: 0`, `tempo`, `loudness`, `time_signature`) from the endpoint
is removed from the dataset before training. `song_title` and `artist`
were removed to avoid artist bias, that is one artist being
over-represented in the dataset and having a higher weight than musical
features. `energy` is a little strange. “Typically, energetic tracks
feel fast, loud, and noisy. For example, death metal has high energy,
while a Bach prelude scores low on the scale. Perceptual features
contributing to this attribute include dynamic range, perceived
loudness, timbre, onset rate, and general entropy.”[7]

## The Code

<div id="setup">

### Setup

</div>

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

<div id="the-model">

### The model

</div>

The model is then initiated and built. The parameters were figured out
using trial and error. The size and types of layers were first chosen at
random then manually adjusted, with changes that increased accuracy kept
and those lowering it discarded. Accuracy assessment is elaborated on
further down. This model could be vastly improved with more time to
tweak it and more sophisticated tools.

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

Now the model is compiled. Again, parameters were determined by trial
and error. Given more time and resources, these too could be further
optimized. The `Adam` model is described as “computationally efficient,
has little memory requirements, is invariant to diagonal rescaling of
the gradients, and is well suited for problems that are large in terms
of data and/or parameters.”[8] They also say that the parameters require
“little tuning” which is perfect for this experiment. The two metrics,
`accuracy` and `BinaryAccuracy`, are used within our loss function
`binary_crossentropy`. These metrics are what we seek to maximize while
minimizing the loss function. `EarlyStopping` is used to prevent the
model from training endlessly while very minute changes are occuring
within the `accuracy` and `BinaryAccuracy` metrics. This prevents the
waste of computational resources and iterations on the algorithm to
occur faster. The patience of `50` means that the training stops after
50 epochs of BinaryAccuracy not improving. The `batch_size` and
`validation_split` are simply set to values that worked best without
taking a lot of time to compute. A larger batch size didn’t improve
accuracy and made the model training take significantly longer.

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

This part of the code shows an analysis of the model’s training. The
data here will be explored further in the [training part of the results
section](#training).

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

This part of code is for testing the accuracy of the model. First, some
packages are imported, and a `test` function is defined. This function
takes an `X` and `Y`, which are the inputs and the expected outputs
respectively. The model is then used to make predictions from the inputs
`X`, into a new variable `preds`. These predictions are rounded to
create clean classifications, as the `predict` function returns a float.
Using the predictions `preds` and the actual values `Y`, a confusion
matrix is generated. A confusion matrix shows the types errors made by
the algorithm. Following the confusion matrix, a full classification
report is generated.

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def test(X, Y):
	preds = np.round(model.predict(X),0)
	print(confusion_matrix(Y, preds))
	print(classification_report(Y, preds))
```


Tests are then conducted on the testing data, the training data, and the
whole dataset.

```python
test(X_test, Y_test)
test(X_train, Y_train)
test(X,Y)
```
<div id="results">

# Results

</div>

## Training

## Testing Results

<div id="under-500-epochs">

### Under 500 Epochs

</div>

<div id="Dataset Classification Report">

|              | precision |   recall | f1-score |     support |
|:-------------|----------:|---------:|---------:|------------:|
| Disliked     |  0.677713 | 0.820461 | 0.742287 |  997.000000 |
| Liked        |  0.779012 | 0.618627 | 0.689617 | 1020.000000 |
| accuracy     |  0.718394 | 0.718394 | 0.718394 |    0.718394 |
| macro avg    |  0.728363 | 0.719544 | 0.715952 | 2017.000000 |
| weighted avg |  0.728940 | 0.718394 | 0.715652 | 2017.000000 |

Classification Report of the Overall Data

</div>

<div id="Training Data Classification Report">

|              | precision |   recall | f1-score |    support |
|:-------------|----------:|---------:|---------:|-----------:|
| Disliked     |  0.688196 | 0.829530 | 0.752282 |  745.00000 |
| Liked        |  0.793160 | 0.634941 | 0.705286 |  767.00000 |
| accuracy     |  0.730820 | 0.730820 | 0.730820 |    0.73082 |
| macro avg    |  0.740678 | 0.732236 | 0.728784 | 1512.00000 |
| weighted avg |  0.741441 | 0.730820 | 0.728442 | 1512.00000 |

Classification Report of the Training Data

</div>

<div id="Test Data Classification Report">

|              | precision |   recall | f1-score |    support |
|:-------------|----------:|---------:|---------:|-----------:|
| Disliked     |  0.647249 | 0.793651 | 0.713012 | 252.000000 |
| Liked        |  0.734694 | 0.569170 | 0.641425 | 253.000000 |
| accuracy     |  0.681188 | 0.681188 | 0.681188 |   0.681188 |
| macro avg    |  0.690972 | 0.681410 | 0.677219 | 505.000000 |
| weighted avg |  0.691058 | 0.681188 | 0.677148 | 505.000000 |

Classification Report of the Test Data

</div>

<div id="over-1000-epochs">

### Over 1000 Epochs

</div>

<div id="1994 Epoch Dataset Classification Report">

|              | precision |   recall | f1-score |     support |
|:-------------|----------:|---------:|---------:|------------:|
| Disliked     |  0.731903 | 0.821464 | 0.774102 |  997.000000 |
| Liked        |  0.801782 | 0.705882 | 0.750782 | 1020.000000 |
| accuracy     |  0.763014 | 0.763014 | 0.763014 |    0.763014 |
| macro avg    |  0.766843 | 0.763673 | 0.762442 | 2017.000000 |
| weighted avg |  0.767241 | 0.763014 | 0.762309 | 2017.000000 |

Classification Report of the Overall Data After 1994 epochs

</div>

<div id="1994 Epoch Test Data Classification Report">

|              | precision |   recall | f1-score |    support |
|:-------------|----------:|---------:|---------:|-----------:|
| Disliked     |  0.680556 | 0.777778 | 0.725926 | 252.000000 |
| Liked        |  0.741935 | 0.636364 | 0.685106 | 253.000000 |
| accuracy     |  0.706931 | 0.706931 | 0.706931 |   0.706931 |
| macro avg    |  0.711246 | 0.707071 | 0.705516 | 505.000000 |
| weighted avg |  0.711306 | 0.706931 | 0.705476 | 505.000000 |

Classification Report of the Test Data After 1994 epochs

</div>

<div id="1994 Epoch Training Data Classification Report">

|              | precision |   recall | f1-score |     support |
|:-------------|----------:|---------:|---------:|------------:|
| Disliked     |  0.749699 | 0.836242 | 0.790609 |  745.000000 |
| Liked        |  0.820852 | 0.728814 | 0.772099 |  767.000000 |
| accuracy     |  0.781746 | 0.781746 | 0.781746 |    0.781746 |
| macro avg    |  0.785275 | 0.782528 | 0.781354 | 1512.000000 |
| weighted avg |  0.785793 | 0.781746 | 0.781220 | 1512.000000 |

Classification Report of the Training Data After 1994 epochs

</div>

<div id="discussion-and-analysis">

# Discussion and Analysis

</div>

## The model

The model’s accuracy over all the testing configurations never exceeds
72%. This accuracy score isn’t particularly impressive and could
potentially be improved with by tweaking the model. Despite this low
accuracy, this model could still be deployed and would function
adequately. Looking at the results, the rate at which the model predicts
the user will like a song when they actually wouldn’t is \~9%, which is
half as frequently as the rate at which it it predicts the user would
dislike a song when they would actually like it (\~19%). This is a good
thing from a curation perspective. It’s a better that most of the time,
the user isn’t suggested tracks that they won’t like.

These results line up with the experiment presents by the dataset
author. Their model was able to get to an accuracy of about 75%.[9]

## How it could improved

This system could be improved with a much larger model that takes into
account more factors. Taking artists into account, for example, might
help improve accuracy, but that was outside of the scope of this
project. Furthermore, having more user data to corroborate with. If user
A’s preferences are similar to user B’s, one could use user A *and* user
B’s data to predict whether or not user B will like a song.
Agglomerating user data would also help cover any missing data in a
single user. More data would also help. This algorithm was only trained
on 2017 pieces of music, while Spotify reports having “over 70 million
tracks.”[10] The dataset size being so small was a practical problem,
acquiring data on users’ listening preferences is difficult without
being Spotify itself; the data in the dataset was manually collected by
the creator. Manual data collection makes this process very time
consuming and increases the difficulty of scaling. Further training on
the current data won’t improve the model. The consequences of further
training on the current data (> 1000 epochs) varies from marginal
improvements to overfitting, where the improvements are substantial but
only occur when predicting using the training data. The model performs
about the same on testing data when trained for 500 epochs and more than
1000 epochs. The accuracy of the model goes from 0.68 with under 500
epochs to 0.71 with close to 2000 epochs, a 3% increase with 4× the
computation time.

## What Spotify Actually Does

Spotify not only benefits from a much larger dataset (over 70 million
tracks) and many more users, they also benefit from more varied data.
Spotify tracks your listening history, when you skip songs, how you
organize your playlist, as well as where you are when you listen to a
track.[11] Spotify’s massive user base allows them to group people with
similar listening habits together.[12] They also deploy varied
algorithms that analyze not only user and the music, but playlists
themselves (using Natural Language Processing).[13]

<div id="conclusions">

# Conclusions

</div>

This experiment demonstrates that it is possible to recommend music
using a Neural Network. While the accuracy of the model leaves a little
to be desired, this proof of concept works despite only having a
relatively small dataset that only includes information about the music
itself. Major music streaming services have much larger and more varied
datasets to train their models on using both Neural Network and more
sophisticated techniques such as Natural Language Processing showing
that this approach is not only viable, but profitable.

<div id="bibliography">

# Bibliography

</div>

<div id="refs">

</div>

pre<span id="ref-Balaganur2020"></span> Balaganur, Sameer. “How
Spotify’s Algorithm Manages to Find Your Inner Groove.” Analitics India
Magazine, January 2020.
<https://analyticsindiamag.com/how-spotifys-algorithm-manages-to-find-your-inner-groove/>.

pre<span id="ref-sklearn_api"></span> Buitinck, Lars, Gilles Louppe,
Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad
Niculae, et al. “API Design for Machine Learning Software: Experiences
from the Scikit-Learn Project.” In *ECML PKDD Workshop: Languages for
Data Mining and Machine Learning*, 108–22, 2013.

pre<span id="ref-Irvine2019"></span> Irvine, Dr. Martin. “Music to My
Ears: De-Blackboxing Spotify’s Recommendation Engine,” 2019.
<https://blogs.commons.georgetown.edu/cctp-607-spring2019/2019/05/06/music-to-my-ears-de-blackboxing-spotifys-recommendation-algorithm/>.

pre<span id="ref-KerasDense"></span> Keras. “Dense Layer.” *Keras API
Reference*, n.d. <https://keras.io/api/layers/core_layers/dense/>.

pre<span id="ref-KerasDropout"></span> ———. “Dropout Layer.” *Keras API
Reference*, n.d.
<https://keras.io/api/layers/regularization_layers/dropout/>.

pre<span id="ref-KerasActivation"></span> ———. “Layer Activation
Functions.” *Keras API Reference*, n.d.
<https://keras.io/api/layers/activations/>.

pre<span id="ref-kingma2017adam"></span> Kingma, Diederik P., and Jimmy
Ba. “Adam: A Method for Stochastic Optimization,” 2017.
<https://arxiv.org/abs/1412.6980>.

pre<span id="ref-Marius2021"></span> Marius, Hucker. “Uncovering How the
Spotify Algorithm Works,” November 2021.
<https://towardsdatascience.com/uncovering-how-the-spotify-algorithm-works-4d3c021ebc0>.

pre<span id="ref-McIntire2017"></span> McIntire, George. “A Machine
Learning Deep Dive into My Spotify Data.” *Open Data Science*. ODSC,
June 2017.
<https://opendatascience.com/a-machine-learning-deep-dive-into-my-spotify-data/>.

pre<span id="ref-McIntire"></span> ———. “Spotify Song Attributes.”
*Kaggle*, n.d. <https://www.kaggle.com/geomack/spotifyclassification>.

pre<span id="ref-Nduati2020"></span> Nduati, Judy. “Introduction to
Neural Networks,” October 2020.
<https://www.section.io/engineering-education/introduction-to-neural-networks/>.

pre<span id="ref-scikit-learn"></span> Pedregosa, F., G. Varoquaux, A.
Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, et al.
“Scikit-Learn: Machine Learning in Python.” *Journal of Machine Learning
Research* 12 (2011): 2825–30.

pre<span id="ref-Schmidhuber2015"></span> Schmidhuber, Jürgen. “Deep
Learning in Neural Networks: An Overview.” *Neural Networks* 61 (2015):
85–117. <https://doi.org/10.1016/j.neunet.2014.09.003>.

pre<span id="ref-Spotify2021News"></span> Spotify. “About Spotify.”
*Spotify Newsroom*, December 2021.
<https://newsroom.spotify.com/company-info/>.

pre<span id="ref-Spotify2021Dev"></span> ———. “Web API Reference:
Spotify for Developers.” *Spotify for Developer*. Spotify, 2021.
<https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features>.

<div id="appendix">

# Appendix

</div>

## Source Code

Full repository found [here](https://github.com/shadowlerone/ESP).

## Model Summary

<div id="summary">

| Type    | Shape      | Param # |
|:--------|:-----------|:--------|
| Dense   | (None, 10) | 100     |
| Dense   | (None, 10) | 110     |
| Dense   | (None, 25) | 275     |
| Dropout | (None, 25) | 0       |
| Dense   | (None, 10) | 260     |
| Dense   | (None, 10) | 110     |
| Dense   | (None, 10) | 110     |
| Dense   | (None, 10) | 110     |
| Dense   | (None, 10) | 110     |
| Dense   | (None, 10) | 110     |
| Dropout | (None, 10) | 0       |
| Dense   | (None, 5)  | 55      |
| Dense   | (None, 1)  | 6       |

Summary of the Keras Model

</div>

## Full Code

``` python
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

model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy','BinaryAccuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = EarlyStopping(monitor='val_binary_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=50,
                                   restore_best_weights=True)

# now we just update our model fit call
history = model.fit(X_train,
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

[1] Judy Nduati, “Introduction to Neural Networks,” October 2020,
<https://www.section.io/engineering-education/introduction-to-neural-networks/>.

[2] Ibid.

[3] Keras, “Dropout Layer,” *Keras API Reference*, n.d.,
<https://keras.io/api/layers/regularization_layers/dropout/>.

[4] Keras, “Layer Activation Functions,” *Keras API Reference*, n.d.,
<https://keras.io/api/layers/activations/>.

[5] George McIntire, “Spotify Song Attributes,” *Kaggle*, n.d.,
<https://www.kaggle.com/geomack/spotifyclassification>.

[6] Spotify, “Web API Reference: Spotify for Developers,” *Spotify for
Developer* (Spotify, 2021),
<https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features>.

[7] Ibid.

[8] Diederik P. Kingma and Jimmy Ba, “Adam: A Method for Stochastic
Optimization,” 2017, <https://arxiv.org/abs/1412.6980>.

[9] George McIntire, “A Machine Learning Deep Dive into My Spotify
Data.” *Open Data Science* (ODSC, June 2017),
<https://opendatascience.com/a-machine-learning-deep-dive-into-my-spotify-data/>.

[10] Spotify, “About Spotify,” *Spotify Newsroom*, December 2021,
<https://newsroom.spotify.com/company-info/>.

[11] Sameer Balaganur, “How Spotify’s Algorithm Manages to Find Your
Inner Groove” (Analitics India Magazine, January 2020),
<https://analyticsindiamag.com/how-spotifys-algorithm-manages-to-find-your-inner-groove/>.

[12] Dr. Martin Irvine, “Music to My Ears: De-Blackboxing Spotify’s
Recommendation Engine,” 2019,
<https://blogs.commons.georgetown.edu/cctp-607-spring2019/2019/05/06/music-to-my-ears-de-blackboxing-spotifys-recommendation-algorithm/>.

[13] Ibid.

