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