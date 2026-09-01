import base64
import asyncio
import random
from pathlib import Path
from typing import Annotated

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from alphalibras_app.core.model_inference import SignClassifier
from alphalibras_app.utils.hand_preprocess import extrair_landmarks

router = APIRouter(
    prefix="/api/v1",
    tags=["Classification"]
)

print("Carregando o detector de mãos do MediaPipe...")
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("Detector de mãos carregado.")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "modelo_libras.tflite"
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

classifier = SignClassifier(model_path=str(MODEL_PATH), class_names=CLASSES, confidence_threshold=0.80)

# Lista de palavras para os exercícios de soletração.
# Mantem apenas letras presentes em CLASSES; letras com movimento ou sem suporte
# no modelo atual (como H, J, K, X e Z) ficam fora dos desafios validados.
PALAVRAS_EXERCICIOS = [
    "AMOR",
    "AR",
    "ASA",
    "AVE",
    "BOLA",
    "BOLO",
    "BOM",
    "CASA",
    "COLA",
    "COR",
    "CURA",
    "DADO",
    "DAMA",
    "DEDO",
    "DEUS",
    "DIA",
    "DOCE",
    "DOR",
    "ELA",
    "ELE",
    "ESCOLA",
    "EU",
    "FACA",
    "FILA",
    "FIM",
    "FOGO",
    "GALO",
    "GATO",
    "GELO",
    "GOTA",
    "IDA",
    "IGUAL",
    "LADO",
    "LAMA",
    "LAR",
    "LATA",
    "LEAL",
    "LUA",
    "LUVA",
    "MAR",
    "MALA",
    "MESA",
    "MEU",
    "MIL",
    "MIMO",
    "MINA",
    "MUNDO",
    "MUDO",
    "NADA",
    "NOME",
    "NOVE",
    "OITO",
    "OVO",
    "PAI",
    "PANO",
    "PATO",
    "PESO",
    "POTE",
    "QUE",
    "QUEDA",
    "RATO",
    "REDE",
    "RIO",
    "RODA",
    "ROSA",
    "RUA",
    "SAPO",
    "SETE",
    "SOL",
    "TETO",
    "TIO",
    "UVA",
    "URSO",
    "VACA",
    "VASO",
    "VER",
    "VIDA",
    "VILA",
    "VIVO",
    "VOA",
    "VOVO",
    "VULTO",
    "WEB",
    "YOGA",
    "COMIDA",
]


def palavras_suportadas() -> list[str]:
    """Retorna palavras que podem ser validadas pelo modelo atual."""
    letras_suportadas = set(CLASSES)
    return [
        palavra
        for palavra in sorted(set(PALAVRAS_EXERCICIOS))
        if set(palavra).issubset(letras_suportadas)
    ]

# ROTAS DE EXERCÍCIOS (HTTP)

@router.get("/exercicios/soletracao")
def get_exercicio_soletracao():
    """Obtém um exercício de soletração.

    Returns:
        dict: Dados do exercício com tipo, palavra sorteada e instrução para o
        usuário.
    """
    palavra_selecionada = random.choice(palavras_suportadas())
    return {
        "tipo": "soletracao",
        "palavra": palavra_selecionada,
        "instrucao": f"Use a câmera para soletrar a palavra: '{palavra_selecionada}'"
    }


@router.get("/exercicios/roadmap")
def get_roadmap_soletracao(
    quantidade: Annotated[int, Query(ge=1, le=30)] = 10
):
    """Sorteia um roteiro de palavras para um desafio de soletração."""
    palavras = palavras_suportadas()
    total = min(quantidade, len(palavras))
    selecionadas = random.sample(palavras, total)

    return {
        "tipo": "roadmap_soletracao",
        "quantidade": total,
        "palavras": selecionadas,
        "letras_suportadas": CLASSES,
        "instrucao": "Complete as palavras na ordem sorteada."
    }


@router.get("/exercicios/letras")
def get_letras_suportadas():
    """Lista letras que o modelo atual consegue classificar."""
    return {
        "letras": CLASSES,
        "observacao": "Letras com movimento ou sem classe treinada ficam fora da validação automática."
    }

@router.get("/exercicios/identificacao")
def get_exercicio_identificacao():
    """Obtém um exercício de identificação de sinal.

    Returns:
        dict: Dados do exercício com a letra correta, as opções de resposta e a
        instrução para o usuário.
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
    """Classifica sinais em tempo real por WebSocket.

    Args:
        websocket: Conexão WebSocket usada para receber frames em base64 e
            enviar a letra prevista.

    Returns:
        None: A função mantém a conexão aberta enquanto o cliente estiver
        conectado.
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
        # Se o cliente desconectou (fechou a aba), a conexão já morreu.
        print("Cliente desconectou.")
        
    except Exception as e:
        # Se houve outro erro (ex: bug no código) é feito uma tentativa de fechar a conexão.
        print(f"Ocorreu um erro no WebSocket: {e}")
        try:
            await websocket.close()
        except RuntimeError:
            # Se já estiver fechado, apenas ignora o erro
            pass
