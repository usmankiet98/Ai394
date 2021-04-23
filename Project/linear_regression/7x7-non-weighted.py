import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv( "train.csv" )

A = train.drop( 'label',axis=1 )

B = train['label']

filter = np.array( [[1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]] )

A = A.to_numpy(  ) 

print( A.shape )

size = np.empty( ( 0,484, int )

def complaxity( image, filter ):

  filter_a, filter_b = filter.shape 
  
  filter_len_by_2 = ( filter_a//2 ) 
  
  n = 28
  
  nn = n - ( filter_len_by_2 *2 )

new_image = np.zeros( ( nn,nn ) )

  for i in range( 0,nn ):
  
    for j in range( 0,nn ):
    
      new_image[i][j] = np.sum( image[i:i+filter_a, j:j+filter_b]*filter )//25
      
  return new_image

subset = 500 

for img in A[0:subset,:]:

  image_2d = np.reshape( img, ( 28,28 ) )
  
  len_image = complaxity( image_2d,filter )
 
  len_image_1d = np.reshape( len_image, ( -1,484 ) )
 
  sA= np.append( sA, len_image_1d, axis=0 ) #size

B = B.to_numpy(  )

sB = B[0:subset]

print( sB.shape )

print( sA.shape )

sATrain, sATest, yTrain, yTest = train_test_split( sA,sB,test_size=0.2,random_state=0 )

print( sATest.shape,", ",yTest.shape )

print( sATrain.shape,", ",yTrain.shape )

line = LinearRegression(  )

line.fit( sATrain,yTrain )

line_prediction_b = Line.predict( sATest )

verify_line = metrics.r2_score( yTest,line_prediction_b ) 

print( 'Linear Regression Accuracy 7*7 unweighted' )

print( verify_line )
