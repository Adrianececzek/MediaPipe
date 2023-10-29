import cv2
import mediapipe as mp

pose = mp.solutions.pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

video = cv2.VideoCapture('film4.mp4')

desired_width = 1500
desired_height = 700

original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))  # Odczytaj framerate filmu

new_width = desired_width
new_height = int((original_height / original_width) * new_width)

start_time = 0
Ilosc_ruchow = 0
previous_height = 0
bmc_trajectory = []

climbing_time = 0.0  # Inicjalizuj zmienną do śledzenia czasu wspinaczki
climbing_start_frame = 0  # Inicjalizuj zmienną do śledzenia klatki początku wspinaczki

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

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
                climbing_start_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))  # Zapisz klatkę początku wspinaczki

        previous_height = current_height

        bmc_position = (scaled_bmc_x, scaled_bmc_y)
        bmc_trajectory.append(bmc_position)

        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                                         circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                         circle_radius=2))

    if climbing_start_frame > 0:
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        climbing_time = (current_frame - climbing_start_frame) / frame_rate  # Oblicz czas na podstawie klatek i framerate

    cv2.putText(frame, f'Ilość ruchów: {Ilosc_ruchow}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Czas wspinaczki: {climbing_time:.2f} s', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for i in range(1, len(bmc_trajectory)):
        cv2.line(frame, bmc_trajectory[i - 1], bmc_trajectory[i], (0, 0, 255), 2)

    cv2.imshow('Szkielet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
pose.close()