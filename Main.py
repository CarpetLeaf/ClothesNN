from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
classes = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто', 'Туфли', 'Рубашка', 'Кросовки', 'Сумка', 'Ботинки']

# Подготавливаем данные
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Загружаем обученную сеть
model = load_model('fashion_mnist_dense.h5')

# Номер картинки с элементом одежды
n_rec = random.randint(0, 9999)

plt.imshow(x_test[n_rec].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# Меняем размерность изображения и нормализуем его
x = x_test[n_rec]
x = np.expand_dims(x, axis=0)

# Запускаем распознавание набора данных, на котором обучалась сеть
predictions = model.predict(x)

# Печатаем предполагаемый ответ
predictions = np.argmax(predictions[0])
print("Номер класса:", predictions)
print("Название класса:", classes[predictions])

# Печатаем правильные номер класса и название
label = np.argmax(y_test[n_rec])
print("Правильный номер класса:", label)
print("Правильное название класса:", classes[label])