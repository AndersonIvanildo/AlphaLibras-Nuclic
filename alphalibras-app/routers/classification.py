import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import base64
import asyncio
import random

# Importações do FastAPI e WebSocket
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Importações dos módulos próprios
from core.model_inference import SignClassifier
from utils.hand_preprocess import extrair_landmarks

router = APIRouter(
    prefix="/api/v1",
    tags=["Classification"]
)

#  CARREGAMENTO DO MODELO
print("Carregando o detector de mãos do MediaPipe...")
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("Detector de mãos carregado.")

MODEL_PATH = Path.cwd() / "models" / "modelo_libras.tflite"
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Instancia do classificador de sinais
classifier = SignClassifier(model_path=str(MODEL_PATH), class_names=CLASSES, confidence_threshold=0.80)

# Lista de palavras para os exercícios de soletração
PALAVRAS_EXERCICIOS = ["CASA", "BOLA", "GATO", "AMOR", "SOL", "LUA"]

# ROTAS DE EXERCÍCIOS (HTTP)

@router.get("/exercicios/soletracao")
def get_exercicio_soletracao():
    """
    Rota para obter um exercício de soletração.
    
    Retorna uma palavra aleatória da nossa lista para o usuário soletrar.
    O front-end usará essa palavra para instruir o usuário e, em seguida,
    abrirá uma conexão WebSocket para validar a soletração em tempo real.
    """
    palavra_selecionada = random.choice(PALAVRAS_EXERCICIOS)
    return {
        "tipo": "soletracao",
        "palavra": palavra_selecionada,
        "instrucao": f"Use a câmera para soletrar a palavra: '{palavra_selecionada}'"
    }

@router.get("/exercicios/identificacao")
def get_exercicio_identificacao():
    """
    Rota para um exercício de identificação de sinal.

    Sorteia uma letra correta e mais 3 opções incorretas para criar
    um exercício de múltipla escolha. O front-end pode usar a
    `letra_correta` para exibir a imagem/vídeo do sinal correspondente.
    """
    letra_correta = random.choice(CLASSES)
    
    # Garante que as opções incorretas sejam diferentes da correta
    opcoes_incorretas = random.sample([c for c in CLASSES if c != letra_correta], 3)
    
    opcoes = opcoes_incorretas + [letra_correta]
    random.shuffle(opcoes) # Embaralha as opções
    
    return {
        "tipo": "identificacao",
        "letra_correta": letra_correta,
        "opcoes": opcoes,
        "instrucao": "Qual letra corresponde ao sinal mostrado?"
    }


# ROTA DE CLASSIFICAÇÃO (WEBSOCKET)

@router.websocket("/classify/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Este é o endpoint principal para a classificação em tempo real.
    
    Ele aceita uma conexão WebSocket, recebe frames de vídeo em formato base64,
    processa cada frame usando o modelo de IA e retorna a letra prevista.
    É a rota "base" de funcionamento do seu sistema de reconhecimento.
    """
    await websocket.accept()
    print("Cliente conectado via WebSocket.")

    try:
        while True:
            base64_str = await websocket.receive_text()
            img_bytes = base64.b64decode(base64_str.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # A lógica de processamento e predição
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            hand_data = extrair_landmarks(entrada=results)
            letra_prevista, confianca = classifier.predict(hand_data)

            response = {"letra": None, "confianca": 0.0}
            if letra_prevista is not None:
                response["letra"] = letra_prevista
                response["confianca"] = float(confianca)

            await websocket.send_json(response)
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("Cliente desconectado.")
    except Exception as e:
        print(f"Ocorreu um erro no WebSocket: {e}")
    finally:
        await websocket.close()
        print("Conexão WebSocket fechada.")