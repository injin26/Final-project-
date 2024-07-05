from PIL import Image

test_transform = transforms.Compose([ #preprocess images
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = test_transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        sign_name = labels[predicted.item()]
    return sign_name

test_images = ["TEST/000_1_0001_1_j.png", "TEST/000_1_0005_1_j.png"] #use test images
for img_path in test_images:
    sign_name = predict_image(img_path)
    print(f"Image: {img_path} - Predicted sign: {sign_name}")