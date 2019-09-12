import cv2
import numpy as np

from keras.models import load_model


model = load_model("base_v2.model")


img = cv2.imread("../img.png")
img = cv2.resize(img,dsize=(50,50))
img = np.reshape(img,[1,50,50,3])

classes = model.predict_classes(img)

print(classes)


