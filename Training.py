from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# В Keras встроены средства работы с популярными наборами данных
# (x_train, y_train) - набор данных для обучения
# (x_test, y_test) - набор данных для тестирования
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Список с названиями классов
classes = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто', 'Туфли', 'Рубашка', 'Кросовки', 'Сумка', 'Ботинки']

# Просматриваем примеры изображений
plt.figure(figsize=(10, 10))
for i in range(100, 150):
    plt.subplot(5, 10, i-100+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])
plt.show()

# Преобразование размерности данных в наборе
# 60000 изображений, 784 пикселя в каждом
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Нормализация данных
x_train = x_train / 255
x_test = x_test / 255

# Просмматриваем пример правильного ответа
n = 0
print(y_train[n])

# Преобразуем метки в формат one hot encoding
# Если не сделать, то выводится будет в формате [1, 0, ... 0]
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
print(y_train[n])

# Описываем архитектуру нейронной сети
# Создаем последовательную модель
model = Sequential()
# Входной полносвязный слой, 800 нейронов (можно менять), 784 входа в каждый нейрон
# Ф-ия активации `rule` широко используется при кол-ве классов > 2
model.add(Dense(800, input_dim=784, activation="relu"))
# Выходной полносвязный слой, 10 нейронов (по количеству классов)
# Ф-ия `softmax` позволяет удобнее работать с вероятностями
model.add(Dense(10, activation="softmax"))

# Компилируем сеть
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())
# input()
# Обучаем нейронную сеть
model.fit(x_train, y_train, batch_size=200, epochs=40, validation_split=0.2, verbose=1)

model.save('fashion_mnist_dense.h5')  # Сохраняем нейронную сеть для последующего использования
scores = model.evaluate(x_test, y_test, verbose=1)  # Проверка качества работы на наборе данных для тестирования
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))

# Проверяем качество распознавания
# Просматриваем пример изображения
# Меняем `n` для просмотра результатов расповзнавания других изображений
n_rec = 5
plt.imshow(x_test[n_rec].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# Меняем размерность изображения и нормализуем его
x = x_test[n_rec]
x = np.expand_dims(x, axis=0)

# Запускаем распознавание набора данных, на котором обучалась сеть
predictions = model.predict(x)

# Данные на выходе сети
print(predictions)

predictions = np.argmax(predictions[0])
print("Номер класса:", predictions)
print("Название класса:", classes[predictions])

# Печатаем правильные номер класса и название
label = np.argmax(y_test[n_rec])
print("Правильный номер класса:", label)
print("Правильное название класса:", classes[label])

print("gg")
