import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta

# Чтобы не кидал ошибку при компилировании
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
import matplotlib.pyplot as plt
images = x_train[:10]
labels = y_train[:10]
num_row = 2
num_col = 5
fig, axes = plt.subplots(num_row, num_col) 
for i in range(10):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

num_classes = 10 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) 
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = np_utils.to_categorical(y_train, num_classes) 
y_test = np_utils.to_categorical(y_test, num_classes)
print('Размер обучающей выборки:', x_train.shape[0])
print('Размер тестовой выборки:', x_test.shape[0])
model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1))) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizer_v1.Adadelta(), metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=200, epochs=25, validation_split=0.2, validation_data=(x_test, y_test))
round(model.evaluate(x_test, y_test)[1], 2)

