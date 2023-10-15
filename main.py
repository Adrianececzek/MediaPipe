import cv2
import mediapipe as mp

pose = mp.solutions.pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

video = cv2.VideoCapture('filmik3.mp4')

new_width = 840
new_height = 960

start_time = 10

video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

Ilosc_ruchow = 0  # Licznik podskoków
previous_height = 0

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break


    frame = cv2.resize(frame, (new_width, new_height))
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Poprawiony nawias

    if results.pose_landmarks:
        # Wykrycie wysokości kluczowego punktu (np. nosa) lub innego punktu charakterystycznego
        current_height = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].y

        # Sprawdzanie, czy nastąpił podskok
        if previous_height > current_height:
            Ilosc_ruchow += 1

        previous_height = current_height

        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                                         circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                         circle_radius=2))

    cv2.putText(frame, f'Ilosc ruchow: {Ilosc_ruchow}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Szkielet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
pose.close()