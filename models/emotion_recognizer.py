import torch
import subprocess
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms.v2 as VT
from typing import Union, Dict
from pydub import AudioSegment
import numpy as np


class EmotionRecognizer(torch.nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        checkpoints_loading_order = ['best', 'last']

        # Create model
        model = AuViLSTMModel(mode='visual', num_classes=8)
        model_name = f"visual_fold{0}"
        load_model(model_name, model, loading_order=checkpoints_loading_order,
                   checkpoints_dir="/home/naif/projects/videoEmotionRecognition/experiments/20241203_011032_sequential_training/model_checkpoints")
        self.temp_dir = "/home/naif/projects/videoEmotionRecognition/temp_frame"
        self.model = model.to(device)
        self.device = device
        face_size = 224
        scale_factor = 1.3

        # Image preprocessing transforms
        self.image_transforms = VT.Compose([
            VT.ToPILImage(),
            VT.ToImage(),
            VT.Resize((224, 224)),
            VT.Grayscale(num_output_channels=3),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=int(face_size * (scale_factor - 1) / 2),
            device=device,
            select_largest=True,
            post_process=False,
            keep_all=False
        )

        # Emotion labels
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    def extract_frames(self, video_path: str, fps: int = 5):
        output_dir = Path(self.temp_dir)
        output_dir.mkdir(exist_ok=True)
        subprocess.call(
            ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps},scale=256:256", f"{output_dir.absolute()}/%03d.png"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        return sorted(output_dir.glob("*.png"))

    def preprocess_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        frame_paths = self.extract_frames(video_path)
        frames = []
        for frame_path in frame_paths:
            try:
                img = Image.open(frame_path)
                face_img = self.mtcnn(img)
                if face_img is not None:
                    if isinstance(face_img, torch.Tensor):
                        face_img = self.image_transforms(face_img.to(torch.uint8))
                    frames.append(face_img)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")

        if not frames:
            raise ValueError("No faces detected in the video. Please check the video quality.")
        return torch.stack(frames).unsqueeze(0)

    def load_audio(self, video_path: str) -> torch.Tensor:
        audio_path = str(Path(video_path).with_suffix(".wav"))
        subprocess.call(
            ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        return torch.from_numpy(samples)

    def clean_temp_frames(self):
        temp_files = Path(self.temp_dir).glob("*.png")
        for file in temp_files:
            file.unlink()

    def predict_emotion(self, video_path: Union[str, Path]) -> Dict:
        video_frames = self.preprocess_video(video_path)
        audio = self.load_audio(video_path)

        input_dict = {'frames': video_frames.to(self.device), 'audio': audio.to(self.device)}
        with torch.inference_mode():
            logits = self.model.forward(input_dict)
            probabilities = torch.softmax(logits, dim=-1)
            top_k = torch.topk(probabilities, k=3)

        self.clean_temp_frames()

        return {
            'top_emotion': self.emotion_labels[top_k.indices[0][0].item()],
            'probabilities': {
                self.emotion_labels[idx]: prob.item()
                for idx, prob in zip(top_k.indices[0], top_k.values[0])
            }
        }
