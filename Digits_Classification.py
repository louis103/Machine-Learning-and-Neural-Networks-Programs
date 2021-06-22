import tensorflow as tf
from tensorflow import keras
# import keras
import numpy as np
import matplotlib.pyplot as plt

# data = keras.datasets.mnist.load_data()
# print(data)
(X_train,y_train) , (X_test,y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0
# y_train = keras.utils.to_categorical(y_train)

X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)
# print(X_train_flattened.shape)
# print(X_test_flattened.shape)

model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation=tf.nn.sigmoid)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train_flattened,y_train,epochs=5)

model.evaluate(X_test_flattened,y_test)
pred = model.predict(X_test_flattened)
print("The number is :",plt.matshow(X_test[1]))
plt.show()
print("Predicted to be == ",np.argmax(pred[1]))


