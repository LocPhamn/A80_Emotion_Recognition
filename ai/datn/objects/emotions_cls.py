import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image

class EmotionClassifier:
    """Class để phân loại cảm xúc từ khuôn mặt"""

    def __init__(self, weights_path, device='cuda', num_classes=7):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading MobileNetV3-Large model...")
        self.model = models.mobilenet_v3_large(pretrained=False)

        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

        # Load weights
        print(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Xử lý nếu state_dict có 'model_state_dict', 'model' hoặc 'state_dict' key
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        print("🎨 Sử dụng preprocessing cho ảnh RGB (3 kênh)")
        # Normalize cho ảnh RGB (3 channels)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Emotion model loaded on {self.device}")
        print(f"Emotion labels: {self.emotion_labels}")

    def predict(self, face_img):
        """
        Dự đoán cảm xúc từ ảnh khuôn mặt

        Args:
            face_img: numpy array (BGR format from OpenCV)

        Returns:
            emotion_label: str, tên cảm xúc
            confidence: float, độ tin cậy
        """
        try:
           
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # Chuyển sang PIL Image
            pil_img = Image.fromarray(face_rgb)

            # Preprocess
            input_tensor = self.preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

            emotion_label = self.emotion_labels[predicted_idx.item()]
            confidence_value = confidence.item()

            return emotion_label, confidence_value

        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return "unknown", 0.0

    def predict_batch(self, face_imgs):
        """
        Dự đoán cảm xúc cho nhiều khuôn mặt cùng lúc (batch processing)
        
        Args:
            face_imgs: List của numpy arrays (BGR format from OpenCV)
        
        Returns:
            List of (emotion_label, confidence) tuples
        """
        if len(face_imgs) == 0:
            return []
        
        try:
            batch_tensors = []
            for face_img in face_imgs:
                # Chuyển BGR sang RGB
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                # Chuyển sang PIL Image
                pil_img = Image.fromarray(face_rgb)
                
                # Preprocess
                input_tensor = self.preprocess(pil_img)
                batch_tensors.append(input_tensor)
            
            # Stack thành batch
            input_batch = torch.stack(batch_tensors).to(self.device)
            
            # Predict toàn bộ batch 1 lần
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, dim=1)
            
            # Tạo kết quả
            results = []
            for idx, conf in zip(predicted_indices, confidences):
                emotion_label = self.emotion_labels[idx.item()]
                confidence_value = conf.item()
                results.append((emotion_label, confidence_value))
            
            return results
        
        except Exception as e:
            print(f"Error in batch emotion prediction: {e}")
            return [("unknown", 0.0)] * len(face_imgs)
