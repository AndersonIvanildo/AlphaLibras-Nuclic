# Importações necessárias
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import base64
import asyncio

# Importações do FastAPI e WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Importações dos módulos do seu projeto
from core.model_inference import SignClassifier
from utils.hand_preprocess import extrair_landmarks

# ---------------------------------- INICIALIZAÇÃO ----------------------------------
# Cria uma instância do aplicativo FastAPI
app = FastAPI(title="AlphaLibras API", description="API para detecção de LIBRAS em tempo real via WebSocket")

# Configuração do CORS (Cross-Origin Resource Sharing)
# Isso permite que seu front-end (que estará em uma origem diferente)
# se comunique com esta API. Em um ambiente de produção,
# é recomendado restringir as origens permitidas.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)


# ---------------------------------- CONFIGURAÇÃO DO MODELO ----------------------------------
# Carregamento do detector de mãos do MediaPipe.
# Esta instância será reutilizada para todas as conexões.
print("Carregando o detector de mãos do MediaPipe...")
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,      # Modo de vídeo, não imagem estática
    max_num_hands=1,              # Aumenta a performance focando em uma mão
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("Detector de mãos carregado.")

# Define o caminho para o modelo TFLite
MODEL_PATH = Path.cwd() / "models" / "modelo_libras.tflite"

# Lista de classes (letras) que o modelo pode prever
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Instancia o nosso classificador de sinais
# Ele carregará o modelo e estará pronto para fazer as previsões.
classifier = SignClassifier(model_path=str(MODEL_PATH), class_names=CLASSES, confidence_threshold=0.80)


# ---------------------------------- ENDPOINT WEBSOCKET ----------------------------------
# Define o endpoint WebSocket na rota "/ws"
@app.websocket("/ws/classify")
async def websocket_endpoint(websocket: WebSocket):
    """
    Este é o endpoint principal que lida com a comunicação WebSocket.
    Ele aceita uma conexão, recebe frames de vídeo em formato base64,
    processa cada frame e retorna a letra prevista.
    """
    # Aceita a conexão do cliente
    await websocket.accept()
    print("Cliente conectado via WebSocket.")

    try:
        # Loop infinito para manter a conexão aberta e receber dados
        while True:
            # Aguarda o recebimento de uma mensagem (frame de vídeo) do cliente
            # A mensagem é esperada como uma string base64
            base64_str = await websocket.receive_text()

            # Decodifica a string base64 para bytes
            # A string recebida pode conter um prefixo "data:image/jpeg;base64,", que removemos
            img_bytes = base64.b64decode(base64_str.split(',')[1])

            # Converte os bytes da imagem em um array NumPy
            np_arr = np.frombuffer(img_bytes, np.uint8)

            # Decodifica o array NumPy para uma imagem no formato do OpenCV (BGR)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue # Pula para a próxima iteração se o frame for inválido

            # ----------------- Processamento e Predição -----------------
            # Converte o frame de BGR para RGB, que é o formato esperado pelo MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Processa a imagem para detectar as mãos
            results = hands_detector.process(frame_rgb)

            # Extrai os landmarks normalizados da mão detectada
            hand_data = extrair_landmarks(entrada=results)

            # Faz a predição da letra usando o classificador
            letra_prevista, confianca = classifier.predict(hand_data)

            # Prepara a resposta para enviar de volta ao cliente
            response = {
                "letra": None,
                "confianca": 0.0
            }

            # Se uma letra foi prevista com confiança suficiente, atualiza a resposta
            if letra_prevista is not None:
                response["letra"] = letra_prevista
                response["confianca"] = float(confianca)

            # Envia a resposta em formato JSON para o cliente
            await websocket.send_json(response)
            
            # Uma pequena pausa para evitar sobrecarga, pode ser ajustado ou removido
            await asyncio.sleep(0.05)


    except WebSocketDisconnect:
        # Este bloco é executado quando o cliente se desconecta
        print("Cliente desconectado.")

    except Exception as e:
        # Captura outras exceções que possam ocorrer durante o processamento
        print(f"Ocorreu um erro: {e}")

    finally:
        # Garante que a conexão seja fechada corretamente em caso de erro ou desconexão
        # Embora o FastAPI geralmente lide com isso, é uma boa prática.
        await websocket.close()
        print("Conexão WebSocket fechada.")

# ---------------------------------- ROTA RAIZ (OPCIONAL) ----------------------------------
# Uma rota GET simples para a raiz da API, útil para verificar se o servidor está online.
@app.get("/")
def read_root():
    return {"status": "AlphaLibras API está online."}

# ---------------------------------- EXECUÇÃO (PARA DEBUG) ----------------------------------
# O bloco abaixo permite executar o servidor diretamente com `python main.py`
# A maneira recomendada para produção é usar um servidor ASGI como o Uvicorn.
# Exemplo de comando para rodar em produção: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    # Inicia o servidor Uvicorn na porta 8000, com recarregamento automático durante o desenvolvimento
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

