import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

######################################################################
######################   Preprocess Dataset   ########################
######################################################################

fileName = "BTC_ETH_4h.csv"
altName = "ETH"

all_data = pd.read_csv(fileName)

# dropping timestamp column
drop_columns = [5]
all_data.drop(all_data.columns[drop_columns], axis=1, inplace=True)
all_data[altName + "_Candle_Color"] = (all_data[altName + "USDT_Close"] - all_data[altName + "USDT_Open"])
all_data[altName + "_Candle_Color"] = np.where(all_data[altName + "_Candle_Color"] > 0, 1, 0)
all_data[altName + "_Candle_Color"] = all_data[altName + "_Candle_Color"].shift(-1, fill_value=0)
all_data.dropna(inplace=True)

min_max_scaler = MinMaxScaler()
clean_df_scaled = min_max_scaler.fit_transform(all_data)
dataset = pd.DataFrame(clean_df_scaled)

dataset.iloc[:, :-2] = np.where(dataset.iloc[:, :-2] > dataset.iloc[:, :-2].shift(1, fill_value=0), 1, 0)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = np.reshape(np.asarray(x_train), (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(np.asarray(x_test), (x_test.shape[0], 1, x_test.shape[1]))

######################################################################
######################### Building Our Model  ########################
######################################################################
model = Sequential()
model.add(Conv1D(filters=1, kernel_size=1))
model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Conv1D(filters=1, kernel_size=1))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(0.0004)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=2000, validation_split=0.2, callbacks=[es])

######################################################################
########################## Visualizing the results ###################
######################################################################
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(x_test)
# Creates a confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index=['Green', 'Red'],
                     columns=['Green', 'Red'])

plt.figure(figsize=(5.5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Model Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

precision, recall, f_score, support = precision_recall_fscore_support(y_test, predictions)

print("precision: {}".format(precision))
print("recall: {}".format(recall))
print("f_score: {}".format(f_score))
print("support: {}".format(support))

accuracy_boost = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy_boost)