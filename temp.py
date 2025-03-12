from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import os
import sys
import pickle

model = YOLO('models/only_puck_1.pt')
img = cv2.imread('try_image.png')

def callback(image):
    res = model.predict(image)[0]
    return sv.Detections.from_ultralytics(res)
    


if os.path.exists('res.pk1'):
    with open('res.pk1', 'rb') as f:
        res = pickle.load(f)

else:
    slicer = sv.InferenceSlicer(callback=callback)
    res = slicer(img)

    with open('res.pk1', 'wb') as f:
        pickle.dump(res, f)




#print("======================")





annotated_frame = sv.BoundingBoxAnnotator().annotate(
    scene=img.copy(),
    detections=res)
        
cv2.imwrite('res_puck.png', annotated_frame)
#print(res)


res2 = model.predict(img)
res2_det = sv.Detections.from_ultralytics(res2[0])

print(res2_det)
print(len(res2_det))

