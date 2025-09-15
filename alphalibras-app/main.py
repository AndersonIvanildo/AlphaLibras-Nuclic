import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

from utils.hand_preprocess import extrair_landmarks, desenhar_esqueleto_na_imagem, desenhar_esqueleto_mao
from core.model_inference import SignClassifier

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MODEL_PATH = Path.cwd() / "models" / "modelo_completo.keras"
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

classifier = SignClassifier(model_path=MODEL_PATH, class_names=CLASSES, confidence_threshold=0.80)

cap = cv2.VideoCapture(0)
IMG_HEIGHT = 480
print("Iniciando a câmera... Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break

    # Preparação da Imagem
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    # Extrai os landmarks normalizados
    hand_data = extrair_landmarks(entrada=results)
    
    letra_prevista, confianca = classifier.predict(hand_data)
    
    if letra_prevista is not None:
        print(f"Letra Prevista: {letra_prevista} | Confiança: {confianca:.2f}")

    frame_original = frame.copy()
    frame_com_esqueleto = desenhar_esqueleto_na_imagem(frame, results)
    esqueleto_pb = desenhar_esqueleto_mao(hand_data, size=IMG_HEIGHT)
    
    h, w, _ = frame_original.shape
    scale = IMG_HEIGHT / h
    new_w = int(w * scale)
    
    resized_original = cv2.resize(frame_original, (new_w, IMG_HEIGHT))
    resized_com_esqueleto = cv2.resize(frame_com_esqueleto, (new_w, IMG_HEIGHT))

    final_image = np.concatenate((resized_original, resized_com_esqueleto, esqueleto_pb), axis=1)

    cv2.imshow('Visualizacao em Tempo Real', final_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
print("Encerrando...")
cap.release()
cv2.destroyAllWindows()
hands_detector.close()
