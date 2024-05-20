import json

from PIL import Image, ImageDraw
import random


def generate_train(index, train=True):
    x1, y1, x2, y2 = random.randint(1, 99), random.randint(1, 99), random.randint(101, 199), random.randint(101, 199)

    # print(width, height)

    # # Создаем новое изображение (белый фон, размер 300x300)
    image = Image.new("RGB", (200, 200), "black")

    # Создаем объект ImageDraw для рисования на изображении
    draw = ImageDraw.Draw(image)

    # Рисуем прямоугольник
    draw.rectangle((x1, y1, x2, y2), outline="white", width=1)

    if train:
        data = {
            "index": index,
            "path": f"dataset/x_train/{index}.png",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

        image_found = write_on_json("dataset/y_train.json", data)
        if not image_found:
            # Сохраняем изображение
            print(f"create image as index #{index}")
            image.save(f"dataset/x_train/{index}.png")
        else:
            print(f"warning, this image has exists #{index}")
    else:
        data = {
            "index": index,
            "path": f"dataset/x_test/{index}.png",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

        image_found = write_on_json("dataset/y_test.json", data)
        if not image_found:
            # Сохраняем изображение
            print(f"create image as index #{index}")
            image.save(f"dataset/x_test/{index}.png")
        else:
            print(f"warning, this image has exists #{index}")


def write_on_json(file_name, data):
    with open(file_name, "r") as json_file:
        load = json.load(json_file)

    for item in load:
        if item["index"] == data["index"]:
            return True

    load.append(data)

    with open(file_name, "w") as json_file:
        # Записываем данные в файл в формате JSON
        json.dump(load, json_file)

    return False


def create_dataset_x(iteration, train=True):
    for index in range(iteration):
        generate_train(index + 1, train)


create_dataset_x(500, False)
