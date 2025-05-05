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

# Create and load the model
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# Load the state dictionary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('AI/best_model.pth', map_location=device))

# Quantize the model to 8-bit
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

model = model.to('cpu')
model.eval()

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

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
    
    food_items = [
        {'id': i, 'name': name} 
        for i, name in enumerate(class_names)
    ]
    
    if request.method == 'POST':
        image_url = request.form.get('img_url')
        if image_url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; FoodAI/1.0)'}
                response = requests.get(image_url, headers=headers)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img_t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_t)
                    _, predicted = outputs.max(1)
                label_idx = predicted.item()
                prediction = class_names[label_idx]
                del img_t
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                error = f"Could not process image: {e}"
        else:
            error = "Please enter a valid image URL."
    return render_template('index.html', 
                         prediction=prediction, 
                         error=error, 
                         image_url=image_url,
                         food_items=food_items)  # Add this line to pass food_items

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)