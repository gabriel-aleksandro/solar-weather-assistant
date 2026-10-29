"""
app.py
======
Interface de chat Gradio para o Solar Weather Assistant.
Deploy: Render (defina GROQ_API_KEY nas Environment Variables).

Uso local:
    export GROQ_API_KEY="gsk_..."
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from src.rag import build_chain

# --------------------------------------------------------------------------- #
# Inicialização (carregada uma vez no startup do Space)
# --------------------------------------------------------------------------- #

print("Inicializando pipeline RAG...")
INIT_ERROR = ""
try:
    CHAIN = build_chain()
    READY = True
    print("Pipeline pronto.")
except FileNotFoundError as e:
    CHAIN = None
    READY = False
    INIT_ERROR = (
        "Índice ChromaDB não encontrado. Execute `python src/ingest.py` "
        "para indexar os documentos antes de iniciar o app."
    )
    print(f"Erro na inicialização: {e}")
except RuntimeError as e:
    CHAIN = None
    READY = False
    INIT_ERROR = str(e)
    print(f"Erro na inicialização: {e}")
except Exception as e:
    CHAIN = None
    READY = False
    INIT_ERROR = f"Erro inesperado: {e}"
    print(f"Erro na inicialização: {e}")


# --------------------------------------------------------------------------- #
# Lógica do chat
# --------------------------------------------------------------------------- #

EXAMPLES = [
    "O que é TSS e por que é usado em previsão de flares solares?",
    "Qual é a diferença entre flare solar e ejeção de massa coronal (CME)?",
    "Como o HARPNUM é usado para evitar data leakage na validação cruzada?",
    "Qual catálogo foi usado para associar CMEs a regiões ativas no SWAN-SF?",
    "Por que a janela de 72h foi escolhida para o labeling de CMEs?",
    "What are the 18 SHARP parameters used in solar flare prediction?",
    "How does bidirectional LSTM differ from standard LSTM for time-series classification?",
]


def chat(message: str, history: list[list[str]]) -> str:
    if not READY:
        return f"Erro na inicialização do pipeline: {INIT_ERROR}"
    if not message.strip():
        return ""
    try:
        response = CHAIN.invoke(message)
        return response
    except Exception as e:
        return f"Erro ao processar a pergunta: {e}"


# --------------------------------------------------------------------------- #
# Interface Gradio
# --------------------------------------------------------------------------- #

DESCRIPTION = """
## Solar Weather Assistant

Assistente especializado em **fisica solar e clima espacial**, com base nos artigos e documentacao do projeto de IC sobre previsao de flares e CMEs (UNICAMP/PIBIC CNPq).

**Pode perguntar sobre:**
- Flares solares e CMEs — o que sao, como sao previstos
- Parametros SHARP do SDO/HMI e o dataset SWAN-SF
- Arquitetura biLSTM + Atencao usada no modelo de previsao
- Metodologia: GroupKFold por HARPNUM, TSS, threshold otimo
- Catalogo cmesrc (Hernandez et al., 2025) e labeling de CMEs

Responde em **portugues ou ingles** conforme a pergunta.
"""

with gr.Blocks(
    title="Solar Weather Assistant",
    theme=gr.themes.Soft(primary_hue="orange"),
) as demo:

    gr.Markdown("# Solar Weather Assistant")
    gr.Markdown(DESCRIPTION)

    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=450, show_label=False),
        textbox=gr.Textbox(
            placeholder="Faça uma pergunta sobre física solar ou espaço climático...",
            container=False,
            scale=7,
        ),
        examples=EXAMPLES,
        cache_examples=False,
        undo_btn="Desfazer",
        clear_btn="Limpar conversa",
    )

    gr.Markdown(
        "**Fontes**: Liu et al. (2020) · Hernandez et al. (2025) · "
        "SWAN-SF dataset · Documentação do projeto de IC — UNICAMP"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
