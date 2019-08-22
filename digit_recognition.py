import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.utils import to_categorical

test_data = pd.read_csv('Desktop/digit recognizer/test.csv')
training = pd.read_csv('Desktop/digit recognizer/train.csv')
train_data = training.drop(columns=['label'])
train_labels=training['label']
train_labels = to_categorical(train_labels)

train_data = train_data.values.reshape(train_data.shape[0], 28, 28, 1) #reshape the input data
test_data = test_data.values.reshape(test_data.shape[0], 28, 28, 1)

 
model = Sequential() #instantiate a Sequential model
model.add(Conv2D(10,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(10,kernel_size=3,activation='relu'))
model.add(Flatten())

model.add(Dense(30, activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics= ['accuracy'])
model.fit(train_data,train_labels,epochs=10)

#write the results into a dataframe in the expected format
Label = model.predict_classes(test_data)
ImageId = np.array([i+1 for i in range(len(Label))])
print(ImageId)
d = {'ImageId': ImageId, 'Label': Label}
df = pd.DataFrame(data=d)
print(df)
df.to_csv(r'Desktop/digit recognizer/predictions.csv',index=False)