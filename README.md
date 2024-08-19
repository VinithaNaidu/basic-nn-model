# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Developing a neural network regression model entails a structured process, encompassing phases such as data acquisition, preprocessing, feature selection, model architecture determination, training, hyperparameter optimization, performance evaluation, and deployment, followed by ongoing monitoring for refinement.

## Neural Network Model

![dl1](https://github.com/Rama-Lekshmi/basic-nn-model/assets/118541549/ef5c4097-a70a-4004-9d3a-c82b9add0fea)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name : D.Vinitha
### Register Number : 212222230175
```
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MARKSDATA').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'int'})
df = df.astype({'OUTPUT':'int'})
df.head()

X = df[['INPUT']].values
y = df[['OUTPUT']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

marks_data = Sequential([Dense(6,activation='relu'),Dense(7,activation='relu'),Dense(1)])

marks_data.compile(optimizer = 'rmsprop' , loss = 'mse')

marks_data.fit(X_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

marks_data.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

marks_data.predict(X_n1_1)


```
## Dataset Information
![image](https://github.com/user-attachments/assets/586062d6-b56d-4097-89b6-bfb09cafe3c7)



## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/e3761337-a02d-4fc5-84e7-5986870804d8)

### Test Data Root Mean Squared Error
![image](https://github.com/user-attachments/assets/8fa95e87-3811-43cb-a71d-86e6629dfa20)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/1b9298fb-7720-40aa-8652-7352101295ff)



## RESULT
 Thus a neural network regression model is developed for the created dataset.
