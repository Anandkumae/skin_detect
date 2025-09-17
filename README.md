# 💆 Skin Type Scanner with Recommendations

A web-based application to detect skin type from a face image using **AI/Deep Learning** (MobileNetV2), and provide personalized skincare recommendations based on skin type, water intake, BMI, and current skincare routine.

---

## 🛠 Features

- **Face detection** using OpenCV's Haar Cascade.  
- **Skin type classification** using a pre-trained MobileNetV2 model (`oily`, `dry`, `normal`).  
- **Automatic image preprocessing** — users do not need to resize or format images.  
- **Personalized recommendations** based on:
  - Skin type
  - Daily water intake
  - Height & weight (BMI)
  - Skincare routine
- **Streamlit-based UI** — easy to use and interactive.  
- Display **detected faces** with rectangles overlaid.

---

## 📁 Project Structure

skin_scanner_app/
│── app.py # Main Streamlit application
│── models/
│ └── haarcascade_frontalface_default.xml
│── saved_models/
│ ├── skin_classifier_mobilenetv2.h5
│ └── class_labels.txt
│── README.md
│── .gitignore

yaml
Copy code

---

## 💻 Requirements

- Python 3.10+  
- Packages (can be installed via `requirements.txt`):

```bash
pip install streamlit tensorflow opencv-python-headless numpy pillow
Note: Use opencv-python-headless to avoid GUI conflicts in Streamlit.

🚀 How to Run
Clone this repository:

bash
Copy code
git clone https://github.com/YourUsername/skin_detect.git
cd skin_detect
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the URL shown in the terminal (usually http://localhost:8501) to access the app.

🖼 Usage
Upload a face or skin image (jpg, jpeg, png).

Optionally, enter your:

Water intake (ml/day)

Weight (kg)

Height (cm)

Current skincare routine

The app will:

Detect your face(s)

Predict your skin type

Show confidence score

Provide personalized recommendations

⚙️ Notes
Ensure saved_models/skin_classifier_mobilenetv2.h5 and models/haarcascade_frontalface_default.xml are present.

The input image will automatically be resized to match the model input (160x160).

Recommendations are simple, rule-based guidelines and not a substitute for professional dermatological advice.

📌 License
This project is open-source and free to use under the MIT License.

🔗 References
OpenCV Haar Cascades

MobileNetV2 Paper

Streamlit Documentation

yaml
Copy code

---

💡 **Tip:** Add a `requirements.txt` with:

streamlit
tensorflow
opencv-python-headless
numpy
pillow
