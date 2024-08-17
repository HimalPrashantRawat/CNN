import tensorflow as tf
from tensorflow import keras 
from keras import Sequential 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import utils 
import matplotlib.pyplot as plt
import pickle
# label_mapping = {0: "class_a", 1: "class_b", 2: "class_c"}

# generators
train_ds = keras.utils.image_dataset_from_directory(
    directory = 'D:\\CNN\\train',  #'''D:\CNN\train it take \t so \\'''
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256,256)
)
validation_ds= keras.utils.image_dataset_from_directory(
    directory = 'D:\\CNN\\test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256,256)
)
'''
# to get prefered name of image
train_ds= keras.utils.image_dataset_from_directory(
    directory = 'D:\\CNN\\train',  #D:\CNN\train it take \t so \\
    labels = 'inferred',
    batch_size = 32,
    image_size = (256,256)
)
train_ds = train_ds.repeat()    
label_mapping = {0: "class_a", 1: "class_b"}
for images, labels in train_ds:
        string_labels = [label_mapping[int(label)] for label in labels]
    "source venv/Scripts/activate"    
'''
# normalize
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# create CNN model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64, kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128, kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=10,validation_data =validation_ds)

with open('history.pkl', 'rb') as file:
    history = pickle.load(file)

plt.plot(history['accuracy'],color = 'red',label='train')
plt.plot(history['val_accuracy'],color = 'blue',label='validation')
plt.legend()
plt.show()

