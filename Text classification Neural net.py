import tensorflow as tf

from tensorflow import keras
# from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Embedding,GlobalAveragePooling1D,Dense,Dropout

data = keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words=10000)

# print(train_data[0])

word_index = data.get_word_index()

# print(train_data[0])
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value,key) for key,value in word_index.items()])
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=250)
'''
def decode_word(text):
    return " ".join([reversed_word_index.get(i,"?") for i in text])

print(len(train_data[0]) ,len(test_data[0]))

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(Dropout(0.5))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

x_value = train_data[:10000]
x_train = train_data[10000:]
# train_data = train_data[10000:]

y_value = train_labels[:10000]
y_train = train_labels[10000:]
# train_labels = train_labels[10000:]

model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_value,y_value),verbose=1)

print("===model overall accuracy===")

model.evaluate(test_data,test_labels)
model.save("IMDB-REVIEW-MODEL.h5")
print("Model written successfully...")
'''
#making predictions

review = "This movie is very bad i have watched it over 100 times and am sad about it"
new_model = keras.models.load_model("IMDB-REVIEW-MODEL.h5")
def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

predicted_sentence = review_encode(review)
predicted_sentence = keras.preprocessing.sequence.pad_sequences([predicted_sentence],value=word_index["<PAD>"],padding="post",maxlen=250)
predictions = new_model.predict(predicted_sentence)
print(np.argmax(predictions))










