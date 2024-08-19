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
![Screenshot 2024-02-24 235751](https://github.com/Rama-Lekshmi/basic-nn-model/assets/118541549/9f8a04b6-c46d-4dec-b338-e5908b1708f2)


## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-02-25 000052](https://github.com/Rama-Lekshmi/basic-nn-model/assets/118541549/0a16ae8b-d4ea-4e2d-b228-3cc381f5accd)

### Test Data Root Mean Squared Error
![Screenshot 2024-02-25 000303](https://github.com/Rama-Lekshmi/basic-nn-model/assets/118541549/d493a408-97be-4317-8ecb-d7ccf9e35a8d)


### New Sample Data Prediction
![Screenshot 2024-02-25 000343](https://github.com/Rama-Lekshmi/basic-nn-model/assets/118541549/e6720ab4-f0fe-4094-b274-8fb76daab635)


## RESULT
 Thus a neural network regression model is developed for the created dataset.
