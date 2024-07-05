import jetson.inference
import jetson.utils
import pandas as pd

# Load labels
labels_df = pd.read_csv("labels.csv")
labels = dict(zip(labels_df["ClassID"], labels_df["Name"]))

# Load the pre-trained model from jetson-inference
net = jetson.inference.imageNet("googlenet")  # You can use your custom model here

def predict_image(image_path):
    img = jetson.utils.loadImage(image_path)
    class_id, confidence = net.Classify(img)
    sign_name = labels.get(class_id, "Unknown")
    return sign_name, confidence

# Testing with images from the TEST directory
test_images = ["TEST/000_1_0001_1_j.png", "TEST/000_1_0005_1_j.png"]

for img_path in test_images:
    sign_name, confidence = predict_image(img_path)
    print(f"Image: {img_path} - Predicted sign: {sign_name} with confidence: {confidence:.2f}")