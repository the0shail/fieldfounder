import keras
import numpy as np
import matplotlib.pyplot as plt
from test import train, test

x_train, y_train = train()
x_test, y_test = test()

print(x_test.shape)

model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(200, 200, 1)),
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="relu")
])

model.compile(loss=keras.losses.MeanSquaredLogarithmicError(), optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=1)

result = model.predict(x_test)

y_pred_avg = np.sum(result, axis=0)
y_true_avg = np.sum(y_test, axis=0)

print(f"Predicted: {y_pred_avg}")
print(f"Real info: {y_true_avg}")

losses = np.subtract(y_pred_avg, y_true_avg)

print(f"losses: {losses}")

plt.plot(y_pred_avg, color="red")
plt.plot(y_true_avg, color="green")

los = keras.losses.mean_squared_logarithmic_error(y_true_avg, y_pred_avg)
print(f"function los: {los}")
# if (los * 100000) < 1:
#     print("save model")
#     model.save("model/recognition_cubes.keras")

plt.show()

