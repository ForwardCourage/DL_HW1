# DL_HW1
This homework requires three ways of image classification for images of different dog species.

## Python packages
```python
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import metrics
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
```

## Reading file names
```python
with open("train.txt", "r") as f:
  lines = [line.rstrip() for line in f]

files = []

for line in lines:
  s = line.split(" ")
  files.append(s[0])
```
files is now a list of file path%names

Let's use this list to read and convert images into 80 * 80 grayscale images
```python
#Reading file names and adding into array
images = []
i = 0
for file in files:
  img = cv.imread(file, 0)
  img = np.array(cv.resize(img, (80, 80)))

  images.append(img)
  if (i%200)==0:
    print(i)
  i = i + 1

images = np.array(images)
```


Now we get the array of images. However, this takes very long time,so we're gonna save it as an .npy file

```python
with open("train_file.npy", "wb") as f:
  np.save(f, images)
```
We get the .npy file of images.

We can get the whole array just by:
```python
with open("train_file.npy", "rb") as f:
  train = np.load(f)

with open("valid_file.npy", "rb") as f:
  valid = np.load(f)

with open("test_file.npy", "rb") as f:
  test = np.load(f)
```

## CNN Method

First, we use basic keras cnn for classification

```python
network = models.Sequential()
network.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape = (80, 80, 1)))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.4))
network.add(layers.Flatten())
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(50, activation='softmax'))

#for top-5accuracy
def top_acc_5(y_true, y_pred):
  return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

network.compile(
    optimizer = 'rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy', top_acc_5]
)


network.summary()
```
The model is constructed. Now let's reshape the input

```python
images_cnn = train.reshape((63325, 80, 80, 1))
images_cnn = images_cnn.astype("float32")/255#Change the range of value to (0, 1)
images_cnn.shape

label_cnn = to_categorical(labels)
```
We have to get the record of the training process

```python
checkpointer = ModelCheckpoint(filepath='/content/gdrive/MyDrive/weights.best.hdf5', 
                               verbose=1, save_best_only=True)
                               
history = network.fit(images_cnn, label_cnn,
          #batch_size=512,
          epochs=10,
          verbose=1,
          validation_split=0.2,
          callbacks=checkpointer)
          
#Check the metrics callable
print(history.history.keys())
```
The accuracy plot

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_top_acc_5'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid', 'top_5'], loc='upper left')
plt.show()
```

The loss plot

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
```

prediction of valid set

```python
valid_cnn = valid.reshape((450, 80, 80, 1))
valid_cnn = valid_cnn.astype("float32")/255

valid_labels_cnn = to_categorical(valid_labels)

valid_loss, valid_acc, top_5_acc_valid = network.evaluate(valid_cnn, valid_labels_cnn)
print(valid_acc)
print(top_5_acc_valid)
```

prediction of test set

```python
test_cnn = test.reshape((450, 80, 80, 1))
test_cnn = test_cnn.astype("float32")/255

test_labels_cnn = to_categorical(test_labels)

test_loss, test_acc, top_5_acc_test = network.evaluate(test_cnn, test_labels_cnn)
print(test_acc)
print(top_5_acc_test)
```

## MLP Method

Unlike cnn, we flatten the images to 1-dim array
```python
#Flatten the images to 1-dim array
X_train = train.reshape((63325, 6400))
X_train = X_train.astype("float32")/255

#Convert labels to arrays like one-hot encoding
y_train = np.array(train_labels)
y_train = to_categorical(y_train)
```

