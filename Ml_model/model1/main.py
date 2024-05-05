import cv2
import numpy as np
from keras.models import load_model
import time

mask_image_path = r"./mask_1920_1080.png"
video_path = r"./data/parking_1920_1080_loop.mp4"
model = load_model('./model/trained_model.keras')

video_capture = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

def preprocess_for_prediction(roi, target_size=(68, 29)):
    roi_resized = cv2.resize(roi, target_size)
    roi_normalized = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)
    return roi_expanded

def draw_bounding_boxes(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def draw_bounding_boxes_and_predict(frame, mask, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        roi_preprocessed = preprocess_for_prediction(roi)
        prediction = model.predict(roi_preprocessed)
        predicted_class = np.argmax(prediction, axis=1)[0]
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    return frame

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to read frame from video or end of video reached.")
        break

    # Process the frame
    processed_frame = draw_bounding_boxes(frame, mask)

    # Display the frame
    cv2.imshow('Processed Video', processed_frame)

    # frame_with_predictions = draw_bounding_boxes_and_predict(frame, mask, model)
    # cv2.imshow('Video with Parking Slot Occupancy', frame_with_predictions)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
