import json
import numpy as np
from PIL import Image


def array_chunk(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def train():
    y_train, x_train = [], []

    with open("dataset/y_train.json", "r") as json_file:
        load = json.load(json_file)

        for item in load:
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            y_train.append([x1, y1, x2, y2])
            image = Image.open(item["path"])
            pixels = list(image.getdata())
            formated_pixels = []

            for pixel in pixels:
                hell = 0.0
                if pixel[0] > 0:
                    hell = pixel[0] / 255

                formated_pixels.append(hell)

            # print(formated_pixels)
            formated_pixels = array_chunk(formated_pixels, 200)

            x_train.append(formated_pixels)
            image.close()

    y_train = np.array(y_train)
    x_train = np.expand_dims(np.array(x_train), axis=3)

    return [x_train, y_train]


def test():
    y_test, x_test = [], []

    with open("dataset/y_test.json", "r") as json_file:
        load = json.load(json_file)

        for item in load:
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            y_test.append([x1, y1, x2, y2])
            image = Image.open(item["path"])
            pixels = list(image.getdata())
            formated_pixels = []

            for pixel in pixels:
                hell = 0.0
                if pixel[0] > 0:
                    hell = pixel[0] / 255

                formated_pixels.append(hell)

            # print(formated_pixels)
            formated_pixels = array_chunk(formated_pixels, 200)

            x_test.append(formated_pixels)
            image.close()

    y_test = np.array(y_test)
    x_test = np.expand_dims(np.array(x_test), axis=3)

    return [x_test, y_test]
