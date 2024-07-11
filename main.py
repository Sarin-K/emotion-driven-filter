import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
from torchvision import transforms

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=4)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss

        return logits


model = FaceModel()
model.load_state_dict(torch.load('best_weights_4_emo.pt', map_location=torch.device('cpu')))
model.eval()


def preprocess_image(frame):
    # Adjust this function to match your model's expected input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((256, 256)),
        transforms.Resize((48, 48)),  # Example: Resize to 48x48
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(frame).unsqueeze(0)


def apply_filter(frame, emotion):
    if emotion == 'happy':
        overlay = frame.copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2YUV)
        overlay[:, :, 0] = cv2.add(overlay[:, :, 0], 20)  # Brighten
        overlay = cv2.cvtColor(overlay, cv2.COLOR_YUV2BGR)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    elif emotion == 'angry':
        # Add dynamic contrast and a deep red overlay for 'angry' emotion
        frame = cv2.addWeighted(frame, 1.15, frame, 0, -50)
        red_overlay = np.full(frame.shape, (0, 0, 255), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.85, red_overlay, 0.15, 0)

    elif emotion == 'sad':
        # Apply a cool, desaturated, and slightly blurred effect for 'sad' emotion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame[:, :, 1] = frame[:, :, 1] * 0.3  # Reduce the saturation more
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        blue_overlay = np.full(frame.shape, (255, 0, 0), dtype=np.uint8)  # Blue overlay
        frame = cv2.addWeighted(frame, 0.85, blue_overlay, 0.15, 0)

    elif emotion == 'fear':
        # Create a vignette effect with high contrast for 'fear'
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=-80)
        # Add a vignette effect to enhance the ominous feel
        rows, cols = frame.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
        vignette = np.dstack([mask] * 3)
        frame = cv2.addWeighted(frame, 1.5, vignette.astype('uint8'), -0.5, 0)

    elif emotion == 'neutral':
        # Apply a subtle sepia effect for 'neutral' emotion
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # sepia_filter = np.array([[0.272, 0.534, 0.131],
        #                          [0.349, 0.686, 0.168],
        #                          [0.393, 0.769, 0.189]])
        # frame = cv2.transform(frame, sepia_filter)
        pass

    return frame


def get_emotion_label(output):
    emotions = ['happy', 'angry', 'fear', 'sad']
    max_prob = torch.max(output).item()
    # May need adjustment
    if max_prob >= 0:
        max_index = torch.argmax(output).item()
        return emotions[max_index]
    return "neutral"


def crop_to_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        return frame[y:y+h, x:x+w]

    return frame


# Main program

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = crop_to_face(frame)
    if not isinstance(cropped_frame, np.ndarray):
        # Convert to ndarray if not already
        cropped_frame = np.array(cropped_frame)
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Get emotion prediction
    with torch.no_grad():
        output = model(preprocessed_frame)
        emotion_label = get_emotion_label(output)

    # Apply filter based on detected emotion and print label
    print(torch.max(output).item())
    print(emotion_label)
    filtered_frame = apply_filter(frame, emotion_label)

    # Display the frame
    cv2.imshow('Emotion-based Filter', filtered_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()

# Add code to detect if there's  face
# Detect face and crop
