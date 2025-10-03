from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import classification

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

app.include_router(classification.router)


@app.get("/")
def read_root():
    """ Rota raiz para verificar se a API está online. """
    return {"status": "AlphaLibras API está online."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)