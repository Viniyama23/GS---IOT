#RM550908 - Vinicius Santos Yamashita ed Farias

import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calcular_distancia(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def braços_cruzados_em_x_acima_da_cabeça(landmarks):
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0

    lw = landmarks[LEFT_WRIST]
    rw = landmarks[RIGHT_WRIST]
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    nose = landmarks[NOSE]

    # Critério 1: ambos os pulsos acima da cabeça (nariz)
    if lw.y < nose.y and rw.y < nose.y:
        # Critério 2: pulsos cruzados
        if lw.x > rs.x and rw.x < ls.x:
            # Critério 3: pulsos próximos (para formar o "X")
            distancia = calcular_distancia(lw, rw)
            if distancia < 0.1:  # Ajuste conforme necessário
                return True
    return False

cap = cv2.VideoCapture(0)

try:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            alerta = False

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark

                if braços_cruzados_em_x_acima_da_cabeça(landmarks):
                    alerta = True
                    # Ajuste do alerta para mais à esquerda
                    cv2.putText(image, 'ALERTA GERADORES LIGADOS!!!', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow('Detecção de Braços Cruzados', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("[INFO] Encerrando o programa com 'q'.")
                break

except KeyboardInterrupt:
    print("\n[INFO] Programa interrompido pelo usuário (CTRL+C).")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados com sucesso.")
