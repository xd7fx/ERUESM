import streamlit as st
from models.emotion_recognizer import EmotionRecognizer  # الاستيراد من الملف الصحيح
from models.auvi_lstm_model import AuViLSTMModel  # تأكد من أن هذا المسار صحيح

print("Starting the Streamlit app...")

MODEL_PATH = "D:\\pro\\rrr-master\\models\\auvi_lstm_model.pkl"

st.title("Emotion Recognition App")

try:
    print("Loading EmotionRecognizer...")
    emotion_recognizer = EmotionRecognizer()
    print("EmotionRecognizer loaded successfully!")
    st.success("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    print("File uploaded successfully!")
    st.video(uploaded_file)
    st.write("Processing video...")

    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        print("Starting prediction...")
        result = emotion_recognizer.predict_emotion(temp_file_path)
        print("Prediction complete!")
        st.write("Top Emotion:", result["top_emotion"])
        st.write("Probabilities:", result["probabilities"])
    except Exception as e:
        print(f"Error processing video: {e}")
        st.error(f"Error processing video: {e}")
