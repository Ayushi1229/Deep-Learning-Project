import cv2
import mediapipe as mp
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import numpy as np


class ProctoringEngine:
    def __init__(self):
        # -------------------- MediaPipe Setup --------------------
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # -------------------- PyTorch Setup (Phone Detection) --------------------
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights)
        self.model.eval()

        # Preprocessing transform for PyTorch
        self.preprocess = self.weights.transforms()

    # -------------------- Phone Detection --------------------
    def check_phone(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(img_rgb).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            class_id = torch.argmax(prob).item()

            # ImageNet class 675 = mobile phone
            if class_id == 675 and prob[class_id] > 0.3:
                return True
        return False

    # -------------------- Eye Gaze Detection --------------------
    def check_eye_gaze(self, landmarks):
        iris_x = landmarks[468].x
        left_corner_x = landmarks[33].x
        right_corner_x = landmarks[133].x

        if iris_x < left_corner_x + 0.005:
            return "Eyes Left"
        if iris_x > right_corner_x - 0.005:
            return "Eyes Right"
        return "Center"

    # -------------------- Head Pose Detection --------------------
    def get_head_pose(self, landmarks):
        nose = landmarks[1]
        forehead = landmarks[10]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        dist_left = abs(nose.x - left_eye.x)
        dist_right = abs(nose.x - right_eye.x)
        dist_up = abs(nose.y - forehead.y)
        dist_down = abs(nose.y - chin.y)

        if dist_left / dist_right > 2.5:
            return "Head Right"
        if dist_right / dist_left > 2.5:
            return "Head Left"
        if dist_up / dist_down > 1.8:
            return "Head Down"
        if dist_down / dist_up > 1.2:
            return "Head Up"
        return "Normal"

    # -------------------- Frame Processing --------------------
    def process_frame(self, frame):
        phone_detected = self.check_phone(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = "STUDENT ON SCREEN"
        color = (0, 255, 0)

        if phone_detected:
            status = "VIOLATION: PHONE DETECTED"
            color = (0, 0, 255)

        elif results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            gaze = self.check_eye_gaze(landmarks)
            pose = self.get_head_pose(landmarks)

            if gaze != "Center" or pose != "Normal":
                status = f"VIOLATION: {gaze} | {pose}".upper()
                color = (0, 0, 255)

        else:
            status = "WARNING: NO FACE DETECTED"
            color = (0, 0, 255)

        return status, color