import pandas as pd
import numpy as np

np.random.seed(1212)
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers

d_f_train = pd.read_csv('train.csv')
d_f_test = pd.read_csv('test.csv')
d_f_train.head()

d_f_features = d_f_train.iloc[:, 1:785]
d_f_label = d_f_train.iloc[:, 0]

X_test = d_f_test.iloc[:, 0:784]

print(X_test.shape)
X_train, X_cv, y_train, y_cv = train_test_split(d_f_features, d_f_label, 
                                                test_size = 0.2,
                                                random_state = 1212)

X_train = X_train.to_numpy().reshape(33600, 784) #(33600, 784)
X_cv = X_cv.to_numpy().reshape(8400, 784) #(8400, 784)

X_test = X_test.to_numpy().reshape(28000, 784)
print((min(X_train[1]), max(X_train[1])))



X_train = X_train.astype('float32'); X_cv= X_cv.astype('float32'); X_test = X_test.astype('float32')
X_train /= 255; X_cv /= 255; X_test /= 255

# Convert labels to One Hot Encoded
digits_count = 10
y_train = keras.utils.to_categorical(y_train, digits_count)
y_cv = keras.utils.to_categorical(y_cv, digits_count)

print(y_train[0]) # 2
print(y_train[3]) # 7

#Features Count
number_input = 784 
one_hidden_numb = 300
two_hidden_numb = 100
three_hidden_numb = 100
four_hidden_numb = 200
digits_count = 10

Inp = Input(shape=(784,))
x = Dense(one_hidden_numb, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(two_hidden_numb, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(three_hidden_numb, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(four_hidden_numb, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(digits_count, activation='softmax', name = "Output_Layer")(x)

model = Model(Inp, output)
model.summary()

learning_rate = 0.1
training_epochs = 20
batch_size = 100
sgd = optimizers.SGD(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history_one = model.fit(X_train, y_train,
                     batch_size = batch_size,
                     epochs = training_epochs,
                     verbose = 2,
                     validation_data=(X_cv, y_cv))

Inp = Input(shape=(784,))
x = Dense(one_hidden_numb, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(two_hidden_numb, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(three_hidden_numb, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(four_hidden_numb, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(digits_count, activation='softmax', name = "Output_Layer")(x)

#ADAM as optimizing methodology
adam = keras.optimizers.Adam(lr=learning_rate)
model2 = Model(Inp, output)

model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])