import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# =============================================
# CLASS LABELS — 35 classes (1-9, A-Z)
# =============================================
class_labels = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
    5: '6', 6: '7', 7: '8', 8: '9',
    9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E',
    14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J',
    19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O',
    24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T',
    29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y',
    34: 'Z'
}

# =============================================
# LOAD MODEL — lightweight, fast on CPU
# =============================================
print("Loading model...")
model = load_model("model_1_aug.h5")
print("Model loaded successfully!")

# =============================================
# WEBCAM SETUP
# =============================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    exit()

print("Webcam started!")
print("Q = Quit | C = Clear text | Space = Add space")

ROI_TOP, ROI_BOTTOM = 100, 400
ROI_LEFT, ROI_RIGHT = 300, 600

sentence = ""
prev_letter = ""
letter_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Draw ROI box
    cv2.rectangle(display_frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
    cv2.putText(display_frame, "Show hand here", (ROI_LEFT, ROI_TOP - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Crop ROI
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

    # Preprocess
    img = cv2.resize(roi, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img, verbose=0)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    predicted_letter = class_labels[predicted_class]

    if confidence > 0.70:
        if predicted_letter == prev_letter:
            letter_count += 1
        else:
            letter_count = 0
            prev_letter = predicted_letter

        if letter_count == 10:
            sentence += predicted_letter
            letter_count = 0

        cv2.putText(display_frame, f"Letter: {predicted_letter}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(display_frame, f"Confidence: {confidence*100:.1f}%",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(display_frame, "No gesture detected",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    # Display sentence
    cv2.putText(display_frame, f"Text: {sentence}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(display_frame, "Q: Quit | C: Clear | Space: Add space",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("ISL Recognition - NIET Project", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
    elif key == ord(' '):
        sentence += " "

cap.release()
cv2.destroyAllWindows()
print("Done!")
