# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Average, BatchNormalization
from keras.callbacks import EarlyStopping

# %%
df = pd.read_csv('data.csv')


# %%
df.info()

# %%
# variable filtering
df.drop(['key'], axis=1, inplace=True)
df.drop(['duration_ms'], axis=1, inplace=True)
df.drop(['mode'], axis=1, inplace=True)
df.drop(['song_title'], axis=1, inplace=True)
df.drop(['artist'], axis=1, inplace=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['tempo'], axis=1, inplace=True)

# %%
df.info()

# %%
# df['Unnamed: 0']

# %%
# split into X and Y
Y = df['target']
X = df.drop(['target'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# print(X.shape)
# print(Y.shape)

# convert to numpy arrays
# X = np.array(X)

# %%
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


# %%
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

# %%
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

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# %%
def test(X, Y):
	preds = np.round(model.predict(X),0)

	# confusion matrix
	print(confusion_matrix(Y, preds)) # order matters! (actual, predicted)

	## array([[490,  59],   ([[TN, FP],
	##       [105, 235]])     [Fn, TP]])

	print(classification_report(Y, preds))

# %%
test(X_test, Y_test)

# %%
test(X_train, Y_train)

# %%
test(X,Y)


