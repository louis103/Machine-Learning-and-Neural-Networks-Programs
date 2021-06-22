from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD,RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2,os
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint

images_path = "C:/Users/LOUIS/Desktop/basedata/training/"
validation_path = "C:/Users/LOUIS/Desktop/basedata/validation/"
test_dir = "C:/Users/LOUIS/Desktop/basedata/testing/"
# img = image.load_img(images_path)
# plt.imshow(img)

# print(cv2.imread(img).shape)
train = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.3
)
test = ImageDataGenerator(
    rescale=1/255
)

train_dataset = train.flow_from_directory(
    directory=images_path,
    target_size=(200,200),
    batch_size=10,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=42,
    subset='training'
)
validation_dataset = train.flow_from_directory(
    directory=validation_path,
    target_size=(200,200),
    batch_size=10,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=42,
    subset='validation'
)
test_images = test.flow_from_directory(
    directory=test_dir,
    target_size=(200, 200),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)
print(test_images[0][0].shape)
Model = Sequential()
Model.add(Conv2D(32,3,activation="relu",input_shape=(200,200,3)))
Model.add(BatchNormalization())


Model.add(Conv2D(64,3,activation="relu"))
Model.add(MaxPooling2D((2)))


Model.add(Conv2D(128,3,activation="relu"))
Model.add(Dropout(0.5))
Model.add(GlobalAveragePooling2D())


Model.add(Flatten())
Model.add(Dense(256,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(3,activation="softmax"))
Model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=['accuracy']
)
filepath="best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

Model.fit(train_dataset,
          validation_data=validation_dataset,
          callbacks=callbacks_list,
          epochs=8)
predictions=Model.evaluate(test_images)
print("LOSS:  " + "%.4f" % predictions[0])
print("ACCURACY:  " + "%.2f" % predictions[1])
predictions = np.argmax(Model.predict(test_images), axis=1)
print(train_dataset.class_indices)
print(predictions)

"""for i in os.listdir(test_dir):
    img = image.load_img(test_dir+'/'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = Model.predict(images)
    print(val)"""


