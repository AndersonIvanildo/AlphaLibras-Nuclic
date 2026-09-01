import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extrair_landmarks(entrada, detector: mp.solutions.hands.Hands = None) -> list | None:
    """
    Extrai os landmarks da mão de uma entrada que pode ser uma imagem ou um
    resultado pré-processado do MediaPipe.
    """
    resultado = None

    if isinstance(entrada, np.ndarray):
        if detector is None:
            raise ValueError("O 'detector' do MediaPipe deve ser fornecido se a entrada for uma imagem.")
        resultado = detector.process(entrada)
    else:
        # A entrada NÃO é uma imagem, então assumi-se que já é um RESULTADO do mediapipe.
        resultado = entrada

    if not resultado.multi_hand_landmarks:
        return None

    # Lógica para encontrar a mão mais próxima
    landmarks_da_mao = None
    profundidade_minima = float('inf')

    for landmarks_da_mao in resultado.multi_hand_landmarks:
        pulso_profundidade = landmarks_da_mao.landmark[0].z

        if pulso_profundidade < profundidade_minima:
            profundidade_minima = pulso_profundidade
            landmarks_mais_proxima = landmarks_da_mao # Armazena as informações de landmarks da mão mais próxima

    if landmarks_mais_proxima is None:
        return None

    # Extração e Normalização das Coordenadas
    vetor_de_landmarks = []

    # Obtenção as coordenadas relativas ao pulso (para invariância de posição)
    posicao_x_pulso = landmarks_mais_proxima.landmark[0].x
    posicao_y_pulso = landmarks_mais_proxima.landmark[0].y
    posicao_z_pulso = landmarks_mais_proxima.landmark[0].z

    for landmark in landmarks_mais_proxima.landmark:
        vetor_de_landmarks.append(landmark.x - posicao_x_pulso)
        vetor_de_landmarks.append(landmark.y - posicao_y_pulso)
        vetor_de_landmarks.append(landmark.z - posicao_z_pulso)

    # Normalização do vetor para ter valores entre -1 e 1 (para invariância de escala)
    valor_absoluto_maximo = max(map(abs, vetor_de_landmarks))
    if valor_absoluto_maximo > 0:
        vetor_normalizado = [coordenada / valor_absoluto_maximo for coordenada in vetor_de_landmarks]
    else:
        # Caso raro onde todos os valores são 0
        vetor_normalizado = vetor_de_landmarks

    return vetor_normalizado


def desenhar_esqueleto_na_imagem(imagem_bgr: np.ndarray, resultado_mediapipe) -> np.ndarray:
    """
    Desenha o esqueleto da(s) mão(s) detectada(s) sobre a imagem original.

    Args:
        imagem_bgr: A imagem original no formato BGR (vinda do OpenCV).
        resultado_mediapipe: O objeto de resultado retornado pela função
                             'hands.process()'.

    Returns:
        Uma nova imagem (cópia da original) com o esqueleto da mão desenhado.
        Se nenhuma mão for detectada, retorna uma cópia da imagem original.
    """
    # Criação de uma cópia da imagem para não modificar a original
    imagem_com_desenho = imagem_bgr.copy()

    # Verificação se alguma mão foi detectada no resultado
    if resultado_mediapipe.multi_hand_landmarks:
        # Iteração sobre cada mão encontrada
        for hand_landmarks in resultado_mediapipe.multi_hand_landmarks:
            # Uso da função pronta do MediaPipe para desenhar os landmarks (pontos) e as conexões (linhas)
            mp_drawing.draw_landmarks(
                image=imagem_com_desenho,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

    return imagem_com_desenho


def desenhar_esqueleto_mao(landmarks_normalizados: list, size=256):
    """
    Cria uma imagem preta com o esqueleto da mão desenhado manualmente com OpenCV
    a partir dos landmarks normalizados.

    Args:
        landmarks_normalizados: Uma lista de 63 floats normalizados (x, y, z)
                                representando os 21 landmarks da mão.
        size: O tamanho (largura e altura) da imagem de saída em pixels.

    Returns:
        Uma imagem NumPy array (size, size, 3) com o esqueleto da mão, ou
        uma imagem preta se os landmarks_normalizados forem None ou inválidos.
    """
    output_image = np.zeros((size, size, 3), dtype=np.uint8) # Fundo preto

    if landmarks_normalizados is None or len(landmarks_normalizados) != 63:
        return output_image

    # Obtenção das conexões padrão do esqueleto da mão do MediaPipe
    connections = mp.solutions.hands.HAND_CONNECTIONS

    # Cálculo das coordenadas em pixels, centralizando e escalonando a mão
    pixel_coords = []

    xs = [landmarks_normalizados[i * 3] for i in range(21)]     # [-> x0 <-, y0, z0, -> x1 <-, y1, z1, ...]
    ys = [landmarks_normalizados[i * 3 + 1] for i in range(21)] # [x0, -> y0 <-, z0, x1, -> y1 <-, z1, ...]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 0.1 # Uma margem para a borda
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Calcula o espaço útil onde poderá ser colocado a imagem
    scale_factor_x = (1 - 2 * padding) / range_x if range_x > 0 else 1
    scale_factor_y = (1 - 2 * padding) / range_y if range_y > 0 else 1
    scale_factor = min(scale_factor_x, scale_factor_y)

    offset_x = padding - min_x * scale_factor
    offset_y = padding - min_y * scale_factor

    for i in range(21):
        x_norm = landmarks_normalizados[i * 3]
        y_norm = landmarks_normalizados[i * 3 + 1]

        px = int((x_norm * scale_factor + offset_x) * size)
        py = int((y_norm * scale_factor + offset_y) * size)
        pixel_coords.append((px, py))

    # Desenhando as linhas (conexões)
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if 0 <= pixel_coords[start_idx][0] < size and \
           0 <= pixel_coords[start_idx][1] < size and \
           0 <= pixel_coords[end_idx][0] < size and \
           0 <= pixel_coords[end_idx][1] < size:
            cv2.line(output_image, pixel_coords[start_idx], pixel_coords[end_idx], (255, 255, 255), 2)

    # Desenhando os pontos (landmarks)
    for point in pixel_coords:
        if 0 <= point[0] < size and 0 <= point[1] < size:
            cv2.circle(output_image, point, 3, (255, 0, 0), -1) # Pontos em azul

    return output_image