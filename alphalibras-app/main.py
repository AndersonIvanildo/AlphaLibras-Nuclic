from utils.hand_preprocess import extrair_landmarks, desenhar_esqueleto_mao, desenhar_esqueleto_na_imagem
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)

# Define a altura padrão para todas as imagens na visualização final
IMG_HEIGHT = 480

print("Iniciando a câmera... Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break

    # 2. Preparação da Imagem
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Processamento ÚNICO por frame
    results = hands_detector.process(frame_rgb)

    # ==============================================================================
    # USO DA NOVA FUNÇÃO NO CONTEXTO DE PRODUÇÃO
    # ==============================================================================

    # 4. Geração das Imagens de Saída
    
    # Imagem 1: O frame original
    frame_original = frame.copy()

    # Imagem 2: Frame com o esqueleto desenhado em cima
    frame_com_esqueleto = desenhar_esqueleto_na_imagem(frame, results)

    # Imagem 3: Esqueleto em fundo preto
    hand_data = extrair_landmarks(entrada=results)
    
    esqueleto_pb = desenhar_esqueleto_mao(hand_data, size=IMG_HEIGHT)

    # ==============================================================================

    # Redimensionamento e Junção
    h, w, _ = frame_original.shape
    scale = IMG_HEIGHT / h
    new_w = int(w * scale)
    
    resized_original = cv2.resize(frame_original, (new_w, IMG_HEIGHT))
    resized_com_esqueleto = cv2.resize(frame_com_esqueleto, (new_w, IMG_HEIGHT))

    # Junta as três imagens horizontalmente
    final_image = np.concatenate((resized_original, resized_com_esqueleto, esqueleto_pb), axis=1)

    # Exibição
    cv2.imshow('Visualizacao em Tempo Real (Original | Com Esqueleto | Apenas Esqueleto)', final_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
print("Encerrando...")
cap.release()
cv2.destroyAllWindows()
hands_detector.close()