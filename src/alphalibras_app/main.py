from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from alphalibras_app.routers import classification

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AlphaLibras API",
    description="API para ensino e detecção de Libras."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(classification.router)


@app.get("/")
def read_root():
    """Serve a interface web da aplicação.

    Returns:
        FileResponse | dict: Arquivo HTML principal quando disponível, ou uma
        resposta JSON informando que a API está online.
    """
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"status": "AlphaLibras API está online, mas index.html não foi encontrado."}


def run():
    """Inicia o servidor ASGI local da aplicação."""
    import uvicorn

    uvicorn.run("alphalibras_app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
