# AlphaLibras-Nuclic

AlphaLibras-Nuclic é uma aplicação para ensino interativo de Libras. O projeto
usa uma API FastAPI, uma interface web servida pelo próprio backend e um modelo
TFLite para reconhecer sinais capturados pela câmera em tempo real.

## Funcionalidades

- Exercício de soletração com palavras sorteadas pela API.
- Classificação de sinais por WebSocket usando frames da câmera.
- Interface web com visualização da câmera e apoio visual para cada letra.
- Script de depuração para testar a detecção em tempo real com OpenCV.

## Estrutura do Projeto

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── src
    └── alphalibras_app
        ├── main.py
        ├── debug.py
        ├── core
        ├── data
        ├── models
        ├── routers
        ├── static
        └── utils
```

## Pré-requisitos

- Python 3.11.
- `uv` instalado.
- Acesso à câmera no navegador.

Para instalar o `uv`, consulte a documentação oficial: https://docs.astral.sh/uv/

## Como Rodar Localmente

Instale as dependências com o `uv`:

```bash
uv sync --python 3.11
```

Inicie a API e a interface web:

```bash
uv run alphalibras-api
```

Outra opção é iniciar diretamente com o Uvicorn:

```bash
uv run uvicorn alphalibras_app.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a aplicação em:

```text
http://localhost:8000
```

## Arquivo requirements.txt

O projeto usa `uv.lock` como arquivo principal de resolução das dependências.
O `requirements.txt` é mantido para compatibilidade com ambientes que ainda
instalam dependências via `pip`.

Para gerar novamente o `requirements.txt` a partir do `uv.lock`, execute:

```bash
uv export --format requirements-txt --output-file requirements.txt --no-hashes --python 3.11
```

Para instalar usando `pip`, quando necessário:

```bash
pip install -r requirements.txt
```

## Depuração com Câmera

Para executar o teste local de detecção com OpenCV:

```bash
uv run python -m alphalibras_app.debug
```

Pressione `q` para encerrar a janela de depuração.
