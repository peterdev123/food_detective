from flask import Flask, request, render_template
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import gc
import os
from functools import lru_cache

# Correct class names from the dataset README
class_names = [
    "Baked Potato", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Sandwich", "Taco", "Taquito",
    "apple_pie", "burger", "butter_naan", "chai", "chapati", "cheesecake", "chicken_curry",
    "chole_bhature", "dal_makhani", "dhokla", "fried_rice", "ice_cream", "idli", "jalebi",
    "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa", "momos", "omelette", "paani_puri",
    "pakode", "pav_bhaji", "pizza", "samosa", "sushi"
]
num_classes = len(class_names)

# Image size limits
MAX_IMAGE_SIZE = 1024  
MAX_FILE_SIZE = 5 * 1024 * 1024  

@lru_cache(maxsize=1)
def load_model():
    """Lazy load the model with caching"""
    # Create and load the model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    # Load the state dictionary
    device = torch.device('cpu')  # Force CPU usage
    model.load_state_dict(torch.load('AI/best_model.pth', map_location=device))

    # Quantize the model to 8-bit
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    
    return model

def process_image(image_data):
    """Process image with size limits and compression"""
    # Open image
    img = Image.open(BytesIO(image_data)).convert('RGB')
    
    if max(img.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0)

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
                # Download image with size limit
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; FoodAI/1.0)'}
                response = requests.get(image_url, headers=headers, stream=True)
                response.raise_for_status()
                
                content_length = int(response.headers.get('content-length', 0))
                if content_length > MAX_FILE_SIZE:
                    raise ValueError(f"Image too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB")
                
                img_t = process_image(response.content)
                
                model = load_model()
                with torch.no_grad():
                    outputs = model(img_t)
                    _, predicted = outputs.max(1)
                label_idx = predicted.item()
                prediction = class_names[label_idx]
                
                # Cleanup
                del img_t
                del outputs
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                error = f"Could not process image: {str(e)}"
        else:
            error = "Please enter a valid image URL"
    return render_template('index.html', 
                         prediction=prediction, 
                         error=error, 
                         image_url=image_url,
                         food_items=food_items)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)