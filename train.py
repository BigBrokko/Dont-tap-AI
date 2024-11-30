from ultralytics import YOLO

# Initialize model
model = YOLO('yolo11n.pt')

model = model.train(data='C:\\Users\\mark2\\Desktop\\Hackerzone\\AI\\Dont-tap-AI\\datasets\\tiles_1\\data.yaml', epochs=100, batch=16)

results = model.val()  # run test set