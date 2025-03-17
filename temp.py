from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import os
import sys
import pickle

model = YOLO('models/only_puck_1.pt')
model2 = YOLO('models/without_puck.pt')
img = cv2.imread('try_image.png')
#img = cv2.imread('with_puck.png')

def callback(image):
    res = model.predict(image)[0]
    return sv.Detections.from_ultralytics(res)
    

'''
if os.path.exists('res.pk1'):
    with open('res.pk1', 'rb') as f:
        res = pickle.load(f)

else:
    slicer = sv.InferenceSlicer(callback=callback)
    res = slicer(img)

    with open('res.pk1', 'wb') as f:
        pickle.dump(res, f)

'''


#print("======================")


res2 = model.predict(img)[0]
res2_det = sv.Detections.from_ultralytics(res2)


print("======================")
#print(res2.boxes)
print("======================")
if len(res2_det):
    print("lkjk")


