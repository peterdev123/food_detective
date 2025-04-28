from flask import Flask, request, render_template
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# Correct class names from your dataset README
class_names = [
    "Baked Potato", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Sandwich", "Taco", "Taquito",
    "apple_pie", "burger", "butter_naan", "chai", "chapati", "cheesecake", "chicken_curry",
    "chole_bhature", "dal_makhani", "dhokla", "fried_rice", "ice_cream", "idli", "jalebi",
    "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa", "momos", "omelette", "paani_puri",
    "pakode", "pav_bhaji", "pizza", "samosa", "sushi"
]
num_classes = len(class_names)

model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('AI/best_model.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    image_url = None
    if request.method == 'POST':
        image_url = request.form.get('img_url')
        if image_url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; FoodAI/1.0)'}
                response = requests.get(image_url, headers=headers)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img_t)
                    _, predicted = outputs.max(1)
                label_idx = predicted.item()
                prediction = class_names[label_idx]
            except Exception as e:
                error = f"Could not process image: {e}"
        else:
            error = "Please enter a valid image URL."
    return render_template('index.html', prediction=prediction, error=error, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)