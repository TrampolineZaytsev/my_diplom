from ultralytics import YOLO

model = YOLO('models/withuout_puck.pt')
results = model.predict('input_video/0304.mp4', save=True)
print("=========================")
for box in results[0].boxes:
    brint(box)