import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception


def detect_humans(net, img):

  bounding_boxes = []
  cropped_people = []

  results = net(img, stream=True)

  for r in results:
      for box in r.boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          w, h = x2-x1, y2-y1

          cls = int(box.cls[0])

          if (cls == 0): #0 == class person
            bounding_boxes.append([x1,y1,w,h])
            cropped_people.append(img[y1:y1+h, x1:x1+w])

  return (cropped_people, bounding_boxes)

def predict_gender(img_list):
  prediction = []
  
  for img_array in img_list:
    cast_img = tf.cast(img_array, tf.float32)
    cast_img = tf.expand_dims(cast_img, 0)
    preprocess = preprocess_inception(cast_img)
    prediction.append(model_inception.predict(preprocess))

  return prediction

def predict_violence(img_array):
  img_array = tf.expand_dims(img_array, 0)
  preprocess = preprocess_mobilenet(img_array)
  prediction = model_mobilenet.predict(preprocess)

  return prediction

def resize(img):
  return tf.image.resize(img, (120,80), tf.image.ResizeMethod.NEAREST_NEIGHBOR)


violence_model_path = 'PATH/PLACE/HOLDER'
gender_model_path = 'PATH/PLACE/HOLDER'

model_mobilenet = tf.keras.saving.load_model(violence_model_path)
model_inception = tf.keras.saving.load_model(gender_model_path)
net = YOLO('../YOLO Weights/yolov9c.pt')

video_path = '/content/7.mp4'
video_cap = cv2.VideoCapture(video_path)

cap_height = np.int_(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_width = np.int_(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
cap_fps = video_cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_out  = cv2.VideoWriter('/content/detection_output_7.mp4', fourcc, cap_fps, (cap_width, cap_height))

font = cv2.FONT_HERSHEY_SIMPLEX

frames = []

violence_count = 0
female_number = 0
max_people_on_screen = 0
frame_count = 0
people_on_screen = []
predictions = []

while video_cap.isOpened():
    ret, frame = video_cap.read()

    if not ret:
      print("Stream end. Exiting.")
      break

    crop, bounding_boxes = detect_humans(net, frame)

    resized = list(map(resize, crop))
    predictions_gender = predict_gender(resized)

    for pred,(x,y,w,h) in zip(predictions_gender, bounding_boxes):

      if(pred[0][0] > pred[0][1]):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Male', (x,y-10), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
      else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Female', (x,y-10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        female_number += 1
        
    resized = tf.image.resize(frame, (120,160), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_cast = tf.cast(resized, tf.float32)
    prediction_violence = predict_violence(resized_cast)

    if(prediction_violence[0][1] > prediction_violence[0][0]):
      violence_count += 1

    for pred in predictions_gender:
        if(pred[0][0] < pred[0][1]):
            female_number += 1

    people_on_screen.append(len(bounding_boxes))
    frame_count += 1

max_people_on_screen = np.argmax(np.bincount(people_on_screen))
has_female = (female_number/max_people_on_screen) > (frame_count/2)
violence_detected = violence_count > 10
#1 = violence; 0 = non-violence
classification = 1 if violence_detected and has_female else 0

video_cap.release()
video_out.release()