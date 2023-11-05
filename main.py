import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math

# Inicjalizacja detektora pozy
pose = mp.solutions.pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

# Inicjalizacja wideo
video = cv2.VideoCapture('film4.mp4')

desired_width = 1500
desired_height = 700

original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))

new_width = desired_width
new_height = int((original_height / original_width) * new_width)

# Inicjalizacja zmiennych do śledzenia
left_arm_angles = []
right_arm_angles = []
bmc_trajectory = []
climbing_time = 0.0
Ilosc_ruchow = 0
previous_height = 0
climbing_start_frame = 0

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

        bmc_x = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].x +
                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x) / 2

        bmc_y = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y +
                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) / 2

        current_height = bmc_y
        scaled_bmc_x = int(bmc_x * new_width)
        scaled_bmc_y = int(bmc_y * new_height)

        if previous_height > current_height:
            Ilosc_ruchow += 1
            if climbing_start_frame == 0:
                climbing_start_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        previous_height = current_height

        bmc_position = (scaled_bmc_x, scaled_bmc_y)
        bmc_trajectory.append(bmc_position)

        if left_shoulder and left_elbow and left_wrist and right_shoulder and right_elbow and right_wrist:
            left_arm_vec = (left_elbow.x - left_shoulder.x, left_elbow.y - left_shoulder.y)
            left_forearm_vec = (left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y)
            left_arm_length = math.hypot(left_arm_vec[0], left_arm_vec[1])
            left_forearm_length = math.hypot(left_forearm_vec[0], left_forearm_vec[1])
            left_angle = math.degrees(
                math.acos((left_arm_vec[0] * left_forearm_vec[0] + left_arm_vec[1] * left_forearm_vec[1]) /
                          (left_arm_length * left_forearm_length)))
            left_arm_angles.append(left_angle)

            right_arm_vec = (right_elbow.x - right_shoulder.x, right_elbow.y - right_shoulder.y)
            right_forearm_vec = (right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y)
            right_arm_length = math.hypot(right_arm_vec[0], right_arm_vec[1])
            right_forearm_length = math.hypot(right_forearm_vec[0], right_forearm_vec[1])
            right_angle = math.degrees(
                math.acos((right_arm_vec[0] * right_forearm_vec[0] + right_arm_vec[1] * right_forearm_vec[1]) /
                          (right_arm_length * right_forearm_length)))
            right_arm_angles.append(right_angle)

        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                                         circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                         circle_radius=2))

    if climbing_start_frame > 0:
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        climbing_time = (current_frame - climbing_start_frame) / frame_rate

    cv2.putText(frame, f'Ilosc ruchww: {Ilosc_ruchow}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Czas wspinaczki: {climbing_time:.2f} s', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for i in range(1, len(bmc_trajectory)):
        cv2.line(frame, bmc_trajectory[i - 1], bmc_trajectory[i], (0, 0, 255), 2)

    cv2.imshow('Szkielet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Po zakończeniu przetwarzania wideo i zamknięciu strumienia wideo

# Plot both left and right arm angles on the same graph with different colors
frames = list(range(len(left_arm_angles)))
plt.plot(frames, left_arm_angles, label='Lewe ramie', color='blue')
plt.plot(frames, right_arm_angles, label='Prawe ramie', color='red')
plt.xlabel('Klatki')
plt.ylabel('Kąt ramion')
plt.title('Wykres kątu ramion')
plt.legend()
plt.show()

# Zamknięcie wszystkich otwartych okien
video.release()
cv2.destroyAllWindows()
pose.close()
