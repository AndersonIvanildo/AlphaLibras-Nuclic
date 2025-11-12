from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import classification

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Criação da instância principal da aplicação
app = FastAPI(
    title="AlphaLibras API",
    description="API para ensino e detecção de LIBRAS."
)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define o caminho para a pasta 'static' que estará no mesmo nível do 'main.py'
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Cria o diretório 'static' se ele não existir
os.makedirs(static_dir, exist_ok=True)

# Monta o diretório 'static' para que o FastAPI sirva arquivos dele
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(classification.router)


@app.get("/")
def read_root():
    """ 
    Rota raiz modificada para servir o seu novo index2.html. 
    """
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "AlphaLibras API está online, mas index2.html não foi encontrado."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
