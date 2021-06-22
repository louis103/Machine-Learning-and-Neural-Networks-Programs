import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from tensorflow import keras

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# plt.matshow(X_train[0])
# plt.show()
X_train = X_train / 255.0
X_test = X_test / 255.0

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten

model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(200, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
model.fit(X_train,y_train,epochs=10)
model.evaluate(X_test,y_test)
plt.matshow(X_test[8])
plt.show()
predictions = model.predict(X_test)

pred = np.argmax(predictions[8])
print("The image was prediced to be in : ",classes[pred])

model.save("FASHION_MNIST_TRAINED_MODEL.h5")

