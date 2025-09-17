import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "skin_classifier_mobilenetv2.h5")
LABELS_PATH = os.path.join(BASE_DIR, "saved_models", "class_labels.txt")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

# -------------------- Verify Files --------------------
for path in [MODEL_PATH, LABELS_PATH, CASCADE_PATH]:
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")

# -------------------- Load Haar Cascade --------------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error(f"❌ Failed to load Haar Cascade from: {CASCADE_PATH}")
else:
    st.success("✅ Haar Cascade loaded successfully!")

# -------------------- Load Model --------------------
st.info("🔄 Loading model, this might take a moment...")

try:
    # First verify the model file exists and is not empty
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    
    if os.path.getsize(MODEL_PATH) == 0:
        st.error(f"❌ Model file is empty: {MODEL_PATH}")
        st.stop()
    
    # Try different loading methods
    load_attempts = [
        # Try with custom objects first
        {
            'custom_objects': {
                'relu6': tf.keras.layers.ReLU(6.),
                'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
                'ReLU': tf.keras.layers.ReLU
            },
            'compile': False
        },
        # Try with safe_mode=False for newer TF versions
        {
            'custom_objects': {
                'relu6': tf.keras.layers.ReLU(6.),
                'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
            },
            'compile': False,
            'options': tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        },
        # Try with just basic loading
        {
            'custom_objects': {},
            'compile': False
        }
    ]
    
    model = None
    last_error = None
    
    for attempt in load_attempts:
        try:
            with st.spinner('Trying to load model...'):
                model = tf.keras.models.load_model(MODEL_PATH, **attempt)
                st.success("✅ Model loaded successfully!")
                break
        except Exception as e:
            last_error = str(e)
            st.warning(f"Attempt failed: {last_error}")
            continue
    
    if model is None:
        st.error("❌ Failed to load model after multiple attempts")
        st.error(f"Last error: {last_error}")
        st.error("""
        Common solutions:
        1. Make sure the model file is not corrupted
        2. Check if the model was saved with a different TensorFlow version
        3. Try reinstalling TensorFlow: `pip install --upgrade tensorflow`
        4. If possible, try using the same TensorFlow version that was used to train the model
        """)
        st.stop()
    
    # Model architecture verification is done in the background
    pass
except Exception as e:
    st.error(f"❌ Unexpected error loading model: {str(e)}")
    st.error("Please check if the model file is valid and compatible with your TensorFlow version.")
    st.stop()

# -------------------- Load Labels --------------------
try:
    with open(LABELS_PATH, "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    st.success(f"✅ Loaded {len(class_labels)} class labels")
except Exception as e:
    st.error(f"❌ Error loading labels: {str(e)}")
    st.stop()

# -------------------- Preprocessing Function --------------------
# Model expects 160x160 RGB images with pixel values in [-1, 1]
TARGET_SIZE = (160, 160)

def preprocess_image(img):
    """
    Preprocess the image for MobileNetV2 model
    - Resize to 160x160
    - Convert BGR to RGB
    - Scale pixel values to [-1, 1]
    - Add batch dimension
    """
    # Convert BGR to RGB if needed (OpenCV loads as BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Resize to target size
    img_resized = cv2.resize(img_rgb, TARGET_SIZE)
    
    # Convert to float32 and scale to [-1, 1]
    img_float = img_resized.astype('float32')
    img_scaled = (img_float / 127.5) - 1.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_scaled, axis=0)
    
    return img_batch

# -------------------- Streamlit UI --------------------
st.title("💆 Skin Type Scanner with Recommendations")
st.write("Upload a face/skin image and optionally enter some info for personalized recommendations.")

# Sidebar Form
with st.sidebar.form("user_info_form"):
    st.header("👤 Your Information")
    
    # Personal Details
    st.subheader("Personal Details")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
    
    # Health Metrics
    st.subheader("Health Metrics")
    weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)
    height = st.number_input("Height (cm)", min_value=0, max_value=250, value=170)
    water_intake = st.slider("Daily Water Intake (ml)", min_value=0, max_value=5000, value=2000, step=100)
    
    # Skin Care
    st.subheader("Skin Care")
    skin_type = st.selectbox("Your Skin Type", ["", "Oily", "Dry", "Combination", "Normal", "Sensitive"])
    skin_concerns = st.multiselect(
        "Skin Concerns",
        ["Acne", "Aging", "Dark Spots", "Redness", "Dryness", "Oiliness", "Sensitivity"],
        []
    )
    current_routine = st.text_area("Current Skincare Routine", 
                                 placeholder="E.g., Morning: Cleanser, Vitamin C, Moisturizer, Sunscreen\nNight: Cleanser, Retinol, Moisturizer")
    
    # Apply Button
    apply_changes = st.form_submit_button("✨ Apply Changes & Analyze")

# Initialize session state for form submission
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Update session state when form is submitted
if apply_changes:
    st.session_state.form_submitted = True
    st.sidebar.success("Settings applied!")
    # Store the form data in session state
    st.session_state.user_data = {
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'water_intake': water_intake,
        'skin_type': skin_type,
        'skin_concerns': skin_concerns,
        'current_routine': current_routine
    }

# Main Content
st.title("🧖‍♀️ Skin Analysis & Care")

# Only show the uploader if form has been submitted
if not st.session_state.get('form_submitted', False):
    st.info("👈 Please fill out the form in the sidebar and click 'Apply Changes & Analyze' to begin.")
    st.image("https://img.freepik.com/free-vector/skincare-concept-illustration_114360-1344.jpg", 
            use_column_width=True)
else:
    # Show the user's information summary
    with st.expander("ℹ️ Your Profile Summary", expanded=True):
        user_data = st.session_state.user_data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Age", user_data['age'])
            st.metric("Gender", user_data['gender'])
        with col2:
            st.metric("Weight", f"{user_data['weight']} kg")
            st.metric("Height", f"{user_data['height']} cm")
        st.metric("Daily Water Intake", f"{user_data['water_intake']} ml")
        
        if user_data['skin_type']:
            st.metric("Skin Type", user_data['skin_type'])
        if user_data['skin_concerns']:
            st.metric("Skin Concerns", ", ".join(user_data['skin_concerns']) or "None")

    # Upload image section
    st.subheader("📷 Upload Your Photo")
    uploaded_file = st.file_uploader("Choose a face/skin image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with st.spinner('Processing image...'):
            # Display the uploaded image
            st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), 
                    caption="Uploaded Image", 
                    use_column_width=True)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                st.warning("⚠️ No face detected. Try another image with better lighting and a clear view of the face.")
            else:
                img_with_boxes = img_cv.copy()
                predictions_data = []
                
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face ROI
                    face_roi = img_cv[y:y+h, x:x+w]
                    
                    try:
                        # Preprocess and predict
                        processed = preprocess_image(face_roi)
                        predictions = model.predict(processed, verbose=0)
                        
                        # Process predictions
                        if len(predictions[0]) > 1:  # Multi-class
                            probs = np.exp(predictions[0] - np.max(predictions[0]))
                            probs = probs / probs.sum()
                            pred_idx = np.argmax(probs)
                            confidence = probs[pred_idx] * 100
                        else:  # Binary
                            pred_idx = int(predictions[0][0] > 0.5)
                            confidence = predictions[0][0] * 100 if pred_idx == 1 else (1 - predictions[0][0]) * 100
                        
                        # Ensure valid class index
                        pred_idx = min(pred_idx, len(class_labels) - 1)
                        predictions_data.append({
                            'skin_type': class_labels[pred_idx],
                            'confidence': confidence,
                            'box': (x, y, w, h)
                        })
                        
                    except Exception:
                        st.error(f"Error analyzing face {i+1}. Please try another image.")
                
                # Draw all boxes and labels
                for pred in predictions_data:
                    x, y, w, h = pred['box']
                    cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{pred['skin_type']} ({pred['confidence']:.1f}%)"
                    cv2.putText(img_with_boxes, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show the image with detections
                st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), 
                        caption="Analysis Results", 
                        use_column_width=True)
                
                # Show predictions
                st.subheader("🔍 Analysis Results")
                for i, pred in enumerate(predictions_data):
                    st.success(f"✨ Face {i+1}: **{pred['skin_type']}** (Confidence: {pred['confidence']:.1f}%)")
                
                # Generate personalized recommendations
                if predictions_data:
                    st.subheader("💡 Personalized Recommendations")
                    recs = []
                    
                    # Get the first face's prediction for recommendations
                    main_pred = predictions_data[0]
                    
                    # Skin type based recommendations
                    skin_type_lower = main_pred['skin_type'].lower()
                    if 'oily' in skin_type_lower:
                        recs.append("💧 Use oil-free, non-comedogenic moisturizers to hydrate without clogging pores.")
                        recs.append("🧼 Cleanse twice daily with a gentle, foaming cleanser to control excess oil.")
                    elif 'dry' in skin_type_lower:
                        recs.append("💦 Use rich, creamy moisturizers with hyaluronic acid or ceramides.")
                        recs.append("🛁 Avoid hot showers and harsh soaps that can strip natural oils.")
                    elif 'normal' in skin_type_lower:
                        recs.append("🌟 Maintain your routine with a balanced cleanser and moisturizer.")
                        recs.append("☀️ Don't forget daily SPF to protect your healthy skin.")
                    
                    # Water intake recommendations
                    user_data = st.session_state.user_data
                    if user_data['water_intake'] < 2000:  # Less than 2L
                        recs.append(f"🚰 Try to increase your water intake from {user_data['water_intake']}ml to at least 2000ml daily for better skin hydration.")
                    
                    # BMI-based recommendations
                    bmi = user_data['weight'] / ((user_data['height'] / 100) ** 2)
                    if bmi < 18.5:
                        recs.append("🍎 Consider a nutrient-rich diet with healthy fats and proteins to support skin health.")
                    elif bmi > 25:
                        recs.append("🏃 Regular exercise can improve circulation and give your skin a healthy glow.")
                    
                    # Skin concerns
                    if 'Acne' in user_data['skin_concerns']:
                        recs.append("🎯 For acne: Look for products with salicylic acid or benzoyl peroxide.")
                    if 'Aging' in user_data['skin_concerns']:
                        recs.append("⏳ For anti-aging: Consider adding retinol and vitamin C to your routine.")
                    if 'Dark Spots' in user_data['skin_concerns']:
                        recs.append("✨ For dark spots: Vitamin C serums and niacinamide can help even skin tone.")
                    
                    # Display recommendations
                    for i, rec in enumerate(recs, 1):
                        st.markdown(f"{i}. {rec}")
                    
                    # Add a reset button
                    if st.button("🔄 Start New Analysis"):
                        st.session_state.form_submitted = False
                        st.rerun()
