import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -------------------------------
# Head Rotation Metrics Function
# -------------------------------
neutral_yaw = None
neutral_pitch = None
neutral_roll = None

def vector_head_pose(landmarks, w, h):

    global neutral_yaw, neutral_pitch, neutral_roll
    
    def p(i):
        return np.array([
            landmarks.landmark[i].x * w,
            landmarks.landmark[i].y * h,
            landmarks.landmark[i].z * w
        ])

    # ---- Stable landmarks ----
    left_eye = p(33)
    right_eye = p(263)
    nose = p(1)
    left_mouth = p(61)
    right_mouth = p(291)

    # ---- Face horizontal axis ----
    eye_vector = right_eye - left_eye
    eye_vector = eye_vector / np.linalg.norm(eye_vector)

    # ---- Face vertical axis ----
    mouth_center = (left_mouth + right_mouth) / 2
    vertical_vector = mouth_center - nose
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)

    # ---- Face normal (looking direction) ----
    face_normal = np.cross(eye_vector, vertical_vector)
    face_normal = face_normal / np.linalg.norm(face_normal)

    # ---- Compute angles ----
    yaw = np.degrees(np.arctan2(face_normal[0], face_normal[2]))
    pitch = np.degrees(np.arctan2(-face_normal[1], face_normal[2])) - 45
    roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

    if neutral_yaw is None:
        neutral_yaw = yaw
        neutral_pitch = pitch
        neutral_roll = roll
        
    yaw -= neutral_yaw
    pitch -= neutral_pitch
    roll -= neutral_roll
    
    forward = (
        abs(yaw) < 18 and
        abs(pitch) < 18 and
        abs(roll) < 15
    )

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "forward": forward
    }    

# -------------------------------
# Status Function
# -------------------------------
def get_status(face_landmarks, w, h):

    # ----- Left Eye -----
    l_iris = face_landmarks.landmark[468]
    l_inner = face_landmarks.landmark[133]
    l_outer = face_landmarks.landmark[33]

    left_ratio = (l_iris.x - l_outer.x) / (l_inner.x - l_outer.x)

    # ----- Right Eye -----
    r_iris = face_landmarks.landmark[473]
    r_inner = face_landmarks.landmark[362]
    r_outer = face_landmarks.landmark[263]

    right_ratio = (r_iris.x - r_inner.x) / (r_outer.x - r_inner.x)

    # ----- Average Gaze -----
    gaze_ratio = (left_ratio + right_ratio) / 2

    # 0.5 means center
    is_gazing_away = not (0.3 < gaze_ratio < 0.6)

    # --- Head Pose ---
    metrics = vector_head_pose(face_landmarks, w, h)

    # --- Center Check ---
    nose_x = face_landmarks.landmark[1].x
    is_off_center = not (0.35 < nose_x < 0.65)

    return is_gazing_away, metrics, is_off_center


# -------------------------------
# Main Loop
# -------------------------------
def main():

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            for landmarks in results.multi_face_landmarks:

                # Draw Mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                # Run Detection
                gaze, metrics, center = get_status(landmarks, w, h)

                # Print angles
                if "yaw" in metrics:
                    print(f"Yaw={metrics['yaw']:.2f} Pitch={metrics['pitch']:.2f} Roll={metrics['roll']:.2f}")

                distracted = gaze or ((not metrics["forward"]) and center)

                if distracted:
                    cv2.putText(frame, "DISTRACTED", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    cv2.putText(frame, "ATTENTIVE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                text = f"Gaze:{gaze} HeadTurn:{not metrics['forward']} OffCenter:{center}"

                cv2.putText(frame, text, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Face Mesh & Attention Detection", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()