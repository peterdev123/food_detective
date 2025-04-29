# 🍔 Food Detective 3000 🍕

A fun, web-based AI app that guesses what kind of food is in your image! Powered by a deep learning model trained on 34 food categories, this project lets users paste an image URL and get an instant, goofy prediction.

---

## 🥗 Features

- **Image URL Input:** Paste any food image URL and get a prediction.
- **34 Food Categories:** From pizza to dal makhani, burgers to samosas.
- **Goofy, User-Friendly UI:** Playful design with emojis and a fun background.
- **Fast AI Predictions:** Powered by a PyTorch ResNet50 model.
- **Runs Locally or Online:** Deployable on your own machine or cloud platforms.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, PyTorch, TorchVision
- **Frontend:** HTML, Tailwind CSS, Jinja2 templates
- **Model:** ResNet50, fine-tuned on the [food_images](https://huggingface.co/datasets/AkshilShah21/food_images) dataset

---

## 📦 Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/yourusername/food-detective-3000.git
    cd food-detective-3000
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Download the trained model:**
    - Place your `best_model.pth` file in the `AI/` directory (or update the path in `app.py`).

4. **Run the app:**
    ```
    python app.py
    ```
    - Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## 🌐 Deployment

You can deploy this app for free on platforms like [Render](https://render.com/), [Heroku](https://heroku.com/), or [PythonAnywhere](https://www.pythonanywhere.com/).

**For Render/Heroku:**
- Make sure to set the host and port in `app.py`:
    ```python
    if __name__ == '__main__':
        import os
        port = int(os.environ.get("PORT", 10000))
        app.run(host='0.0.0.0', port=port)
    ```

---

## 🍕 Food Categories

| Index | Food Name         | Index | Food Name      |
|-------|-------------------|-------|---------------|
| 0     | Baked Potato      | 17    | dhokla        |
| 1     | Crispy Chicken    | 18    | fried_rice    |
| 2     | Donut             | 19    | ice_cream     |
| 3     | Fries             | 20    | idli          |
| 4     | Hot Dog           | 21    | jalebi        |
| 5     | Sandwich          | 22    | kaathi_rolls  |
| 6     | Taco              | 23    | kadai_paneer  |
| 7     | Taquito           | 24    | kulfi         |
| 8     | apple_pie         | 25    | masala_dosa   |
| 9     | burger            | 26    | momos         |
| 10    | butter_naan       | 27    | omelette      |
| 11    | chai              | 28    | paani_puri    |
| 12    | chapati           | 29    | pakode        |
| 13    | cheesecake        | 30    | pav_bhaji     |
| 14    | chicken_curry     | 31    | pizza         |
| 15    | chole_bhature     | 32    | samosa        |
| 16    | dal_makhani       | 33    | sushi         |

---

## 🤖 Credits

- Model trained on [AkshilShah21/food_images](https://huggingface.co/datasets/AkshilShah21/food_images)
- UI inspired by [Tailwind CSS](https://tailwindcss.com/)
- Project by [Peter Vecina](https://github.com/peterdev123)

---

## 🥳 License

MIT License.  
Have fun, and may your snacks always be correctly classified!
