import streamlit as st
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any


class EmotionRecognizer(torch.nn.Module):
    def __init__(self, num_classes: int = 5, hidden_size: int = 384):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_size, num_classes)
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        return self.classifier(h_n[-1])

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        استخراج الميزات من الفيديو.
        """
        print(f"Processing video: {video_path}")

        # افترض ميزات افتراضية للتجربة
        dummy_features = torch.randn(10, self.hidden_size)  # 10 إطارات افتراضية
        return dummy_features.unsqueeze(0)  # [batch_size, sequence_length, hidden_size]

    def predict_emotion(self, video_path: str) -> Dict[str, Any]:
        """
        توقع المشاعر بناءً على ملف الفيديو.
        """
        features = self.preprocess_video(video_path)
        predictions = self.forward(features)
        predicted_class = predictions.argmax(dim=1).item()  # الحصول على الفئة المتوقعة
        
        top_emotion = self.emotion_labels[predicted_class]  # تحويل الفئة إلى شعور
        
        return {
            "top_emotion": top_emotion,
            "probabilities": {
                self.emotion_labels[i]: prob.item()
                for i, prob in enumerate(predictions.softmax(dim=1)[0])  # تعيين الاحتمالات لكل شعور
            }
        }


# إعداد تطبيق Streamlit
st.title("Emotion Recognition App")

# تحميل النموذج
try:
    st.write("Loading EmotionRecognizer...")
    emotion_recognizer = EmotionRecognizer()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# رفع الملف
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processing video...")
    
    try:
        # توقع المشاعر
        result = emotion_recognizer.predict_emotion(uploaded_file.name)
        
        # إظهار النتيجة الأعلى
        st.write("Top Emotion:", result["top_emotion"])
        
        # رسم المخطط الشريطي
        probabilities = result["probabilities"]
        emotions = list(probabilities.keys())
        scores = list(probabilities.values())

        # إعداد المخطط
        fig, ax = plt.subplots()
        ax.bar(emotions, scores, color="skyblue")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotions")
        ax.set_title("Emotion Probabilities")
        st.pyplot(fig)  # عرض المخطط
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
