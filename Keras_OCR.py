import os
import keras_ocr
import matplotlib.pyplot as plt

# pipeline = keras_ocr.pipeline.Pipeline()

# path = "C:\\All data\\IITG\\Projects\\Traffic Light Detection\\Violations"
# dir_list = os.listdir(path)
# print(dir_list)
# for i in range(len(dir_list)):
#     dir_list[i] = "Violations/" + dir_list[i]
    
# print(dir_list)

# images = [
#     keras_ocr.tools.read(img) for img in ["Violations/violation27.jpg"]
# ]

# prediction_groups = pipeline.recognize(images)

# predicted_image = prediction_groups[0]
# for text, box in predicted_image:
#     print(text)

def predict_ocr(img):
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [
    keras_ocr.tools.read(img)
    ]
    prediction_groups = pipeline.recognize(images)
    predicted_image = prediction_groups[0]
    with open("violators.txt", 'a') as file:
        for text, box in predicted_image:
            file.write(text+"\n")

        