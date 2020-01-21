import cv2
import numpy as np
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as pltq
import tensorflow as tf

#Load the saved model
model = tf.keras.models.load_model('model-pro-v2.2.h5')
# Show the model architecture
model.summary()

video = cv2.VideoCapture(0)
arch=pd.read_csv('labels_ecuador.csv')
arch=arch["SignName"].values
while True:
        data=[]
        _, frame = video.read()
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        size_image = im.resize((30, 30))
        data.append(np.array(size_image))
        X_test=np.array(data)
        X_test = X_test.astype('float32')/255 
        pred = model.predict_classes(X_test)
        print(arch[pred[0]])

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()