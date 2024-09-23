from ultralytics import YOLOv10
 
# Load a pretrained YOLOv10n model
model = YOLOv10("mydata/train_v10/weights/best.pt")
 
# Perform object detection on an image
# results = model("test1.jpg")
results = model("mydata/images/val/275.jpg")
print(results)
# results[0].show()