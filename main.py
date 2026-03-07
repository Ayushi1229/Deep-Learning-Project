import cv2
import mediapipe as mp
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# PyTorch model (DL requirement)
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "Proctoring: OK"
    color = (0, 255, 0)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # LEFT EYE
        left_eye_left = landmarks[33].x
        left_eye_right = landmarks[133].x
        left_iris = landmarks[468].x

        # RIGHT EYE
        right_eye_left = landmarks[362].x
        right_eye_right = landmarks[263].x
        right_iris = landmarks[473].x

        # Calculate iris ratio inside the eye
        left_ratio = (left_iris - left_eye_left) / (left_eye_right - left_eye_left)
        right_ratio = (right_iris - right_eye_left) / (right_eye_right - right_eye_left)

        gaze_ratio = (left_ratio + right_ratio) / 2

        if gaze_ratio < 0.35:
            status = "EYE STATUS: LOOKING LEFT"
            color = (0,0,255)

        elif gaze_ratio > 0.65:
            status = "EYE STATUS: LOOKING RIGHT"
            color = (0,0,255)

        else:
            status = "EYE STATUS: LOOKING CENTER"
            color = (0,255,0)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        iris_x = landmarks[468].x
        if iris_x < landmarks[33].x + 0.01:
            status = "EYE STATUS: LOOKING AWAY"
            color = (0, 0, 255)

        nose_x = landmarks[1].x
        if nose_x < 0.4 or nose_x > 0.6:
            status = "HEAD STATUS: LOOKING AWAY"
            color = (0, 0, 255)
    else:
        status = "VIOLATION: NO FACE DETECTED"
        color = (0, 0, 255)

    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("AI Proctor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()