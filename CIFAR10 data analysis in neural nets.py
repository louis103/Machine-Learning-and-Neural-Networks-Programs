import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10

(X_train,y_train),(X_test,y_test) = cifar10.load_data()
classes = ["airplane","automobile","bird","cat","dog","frog","horse","ship","truck"]

y_train = y_train.reshape(-1)

print(X_train.shape)
print(y_train[:5])

#scaling our input data so as to optimize performance of our model
X_train = X_train/255.0
X_test = X_test/255.0
y_test = y_test.reshape(-1)

"""
model = Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(X_train.shape[1:])))
model.add(tf.keras.layers.Dense(3000,activation="relu"))
model.add(tf.keras.layers.Dense(1000,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="sigmoid"))

model.compile(
    optimizer="SGD",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train,y_train,epochs=5)
print("Model evaluation on our test data and test labels")
model.evaluate(X_test,y_test)

"""

#my cnn neural network with 10 layers
model = Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000,activation="relu"))
model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train,y_train,epochs=15)
print("Model evaluation on our test data and test labels")
model.evaluate(X_test,y_test)

predictions = model.predict(X_test)
#making predictions
y_classes = [np.argmax(elem) for elem in predictions]
print(y_classes[:5])
# print(classes[y_classes[0]])
print(y_test[:5])


#8 layers neural network
"""
Epoch 1/5
1563/1563 [==============================] - 80s 44ms/step - loss: 1.5893 - accuracy: 0.4146
Epoch 2/5
1563/1563 [==============================] - 67s 43ms/step - loss: 1.2418 - accuracy: 0.5565
Epoch 3/5
1563/1563 [==============================] - 66s 42ms/step - loss: 1.1057 - accuracy: 0.6092
Epoch 4/5
1563/1563 [==============================] - 60s 39ms/step - loss: 1.0129 - accuracy: 0.6438
Epoch 5/5
1563/1563 [==============================] - 59s 38ms/step - loss: 0.9483 - accuracy: 0.6663
Model evaluation on our test data and test labels
313/313 [==============================] - 4s 12ms/step - loss: 0.9654 - accuracy: 0.6643
[3, 8, 8, 8, 6]
[3 8 8 0 6]
"""







