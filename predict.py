import keras
import numpy as np

from test import test
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

x_test, y_test = test()
# Путь к файлу .h5 с вашей сохраненной моделью
model_path = 'model/recognition_cubes.keras'

# Загрузка модели
loaded_model = keras.models.load_model(model_path)

result = loaded_model.predict(x_test)

y_pred = np.ceil(result[32])
y_true = np.ceil(y_test[32])
print(y_pred)
print(y_true)

image = Image.new("RGB", (50, 50), "white")

draw = ImageDraw.Draw(image)

draw.rectangle(tuple(y_pred), outline="red", width=1)
draw.rectangle(tuple(y_true), outline="green", width=1)

image.show("Predicted")