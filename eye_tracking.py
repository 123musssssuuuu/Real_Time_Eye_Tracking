import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

MIRROR_VIEW = True  # Mirror like selfie

# Open webcam
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mediapipe FaceMesh init
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Start virtual camera
with pyvirtualcam.Camera(width=w, height=h, fps=30) as cam:
    print("✅ Virtual Camera started. Open Zoom/Meet/Teams and select 'PyVirtualCam Camera' as camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw iris points
                for idx in LEFT_IRIS + RIGHT_IRIS:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Show local preview
        cv2.imshow("Eye Tracking (Local Preview)", frame)

        # Send to virtual camera (convert BGR→RGB)
        cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




