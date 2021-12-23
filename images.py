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

from pytablewriter import MarkdownTableWriter

df = pd.read_csv('data.csv')

df.drop(['key'], axis=1, inplace=True)
df.drop(['duration_ms'], axis=1, inplace=True)
df.drop(['mode'], axis=1, inplace=True)
df.drop(['song_title'], axis=1, inplace=True)
df.drop(['artist'], axis=1, inplace=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop(['tempo'], axis=1, inplace=True)

Y = df['target']
X = df.drop(['target'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

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
# model.summary()
table=pd.DataFrame(columns=["Type","Shape", "Param #"])
for layer in model.layers:
    table = table.append({"Type": layer.__class__.__name__,"Shape":layer.output_shape, "Param #": layer.count_params()}, ignore_index=True )

table.to_latex("tables/summary.tex", caption=("Summary of the Keras Model", "Model Summary"), index=False, label="summary", position="H")

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
# plt.rc(usetex=True)
plt.rc('pgf', texsystem='xelatex')

plt.savefig('assets/loss.pgf')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# print(model.predict(X))

def test(X, Y, fp="test", caption=()):
	print(f"Generating graphics for {fp}")
	preds = np.round(model.predict(X),0)
	
	# confusion matrix

	# matrix = MarkdownTableWriter(
	# 	add_index_column=True,
	# 	headers = ["", ""],
	# 	value_matrix = confusion_matrix(Y, preds)
	# )

	disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y, preds, normalize='all'),display_labels=["Disliked", "Liked"])
	disp.plot()
	# plt.rc(usetex=True)
	plt.rc('pgf', texsystem='xelatex')
	plt.savefig(f"assets/{fp}.pgf")
	pd.DataFrame(
				classification_report(Y, preds, output_dict=True,
				target_names=["Disliked", "Liked"]
				)).transpose().to_latex(f"tables/{fp}.tex", caption=caption, label=caption[1], position="H")
	# class_table = MarkdownTableWriter(pd.DataFrame())
	# print()
	

test(X_test, Y_test, "test_data", 
("Classification Report of the Test Data", "Test Data Classification Report"))
test(X_train, Y_train, "training_data", 
	("Classification Report of the Training Data", "Training Data Classification Report")
	)
test(X,Y, "overall_data", ("Classification Report of the Overall Data", "Dataset Classification Report"))

print("Finished regular model rendering")


plt.cla()
plt.clf()

es = EarlyStopping(
	monitor='val_binary_accuracy', 
	mode='max', # don't minimize the accuracy!
	patience=1000,
	restore_best_weights=True
)

history = model.fit(
	X_train,
	Y_train,
	callbacks=[es],
	epochs=1000000, # you can set this to a big number!
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
# plt.rc(usetex=True)
plt.rc('pgf', texsystem='xelatex')

plt.savefig('assets/loss_overfitted.pgf')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# print(model.predict(X))

# def test(X, Y, fp="test", caption=()):
# 	print(f"Generating graphics for {fp}")
# 	preds = np.round(model.predict(X),0)
	

# 	disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y, preds, normalize='all'),display_labels=["Disliked", "Liked"])
# 	disp.plot()
# 	# plt.rc(usetex=True)
# 	plt.rc('pgf', texsystem='xelatex')
# 	plt.savefig(f"assets/{fp}.pgf")
# 	pd.DataFrame(
# 				classification_report(Y, preds, output_dict=True,
# 				target_names=["Disliked", "Liked"]
# 				)).transpose().to_latex(f"tables/{fp}.tex", caption=caption)
# 	# class_table = MarkdownTableWriter(pd.DataFrame())
# 	# print()
	

test(X_test, Y_test, "test_data_overfitted", 
(f"Classification Report of the Test Data After {len(loss_values) + 1} epochs", f"{len(loss_values) + 1} Epoch Test Data Classification Report"))
test(X_train, Y_train, "training_data_overfitted", 
	(f"Classification Report of the Training Data After {len(loss_values) + 1} epochs", f"{len(loss_values) + 1} Epoch Training Data Classification Report")
	)


test(X,Y, "overall_data_overfitted", (f"Classification Report of the Overall Data After {len(loss_values) + 1} epochs", f"{len(loss_values) + 1} Epoch Dataset Classification Report"))


print("Done")