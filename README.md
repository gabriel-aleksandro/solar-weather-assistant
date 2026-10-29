# Solar Weather Assistant

Assistente de Q&A sobre **fisica solar e clima espacial** baseado em RAG (Retrieval-Augmented Generation).

Responde perguntas sobre flares solares, ejecoes de massa coronal (CMEs), parametros SHARP do SDO/HMI e a metodologia de previsao usada no projeto de IC (UNICAMP/PIBIC CNPq), com base nos artigos cientificos e documentacao do projeto.

Demo ao vivo: [solar-weather-assistant.onrender.com](https://solar-weather-assistant.onrender.com)

## Arquitetura

```
Pergunta do usuario
        |
        v
  Embedding (all-MiniLM-L6-v2)
        |
        v
  ChromaDB --> Top-5 chunks relevantes
        |
        v
  Prompt RAG + contexto
        |
        v
  Groq API (LLaMA 3.3 70B)
        |
        v
  Resposta fundamentada nos documentos
```

## Stack

| Componente   | Tecnologia                              |
|--------------|-----------------------------------------|
| LLM          | LLaMA 3.3 70B via Groq API              |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB                                |
| Orquestracao | LangChain                               |
| Interface    | Gradio                                  |
| Deploy       | Render                                  |

## Base de conhecimento

Os documentos indexados cobrem:

- Fisica solar: flares (classificacao GOES), CMEs, SDO/HMI, parametros SHARP
- Dataset SWAN-SF: estrutura, janelas de observacao, desbalanceamento de classes
- Arquitetura do modelo: biLSTM + atencao de Bahdanau, GroupKFold por HARPNUM
- Metodologia: TSS, ensemble, threshold otimo, transfer learning para CME
- Catalogo cmesrc (Hernandez et al., 2025): labeling de CMEs com janela de 72h
- Decisoes de pipeline: por que GroupKFold, por que hold-out, por que Z-score etc.

## Rodando localmente

```bash
git clone https://github.com/gabriel-aleksandro/solar-weather-assistant
cd solar-weather-assistant
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."

# Indexa os documentos (necessario apenas uma vez)
python src/ingest.py

# Inicia o app
python app.py
# Acesse: http://localhost:7860
```

Ou use o script de setup interativo:

```bash
python setup.py
```

## Estrutura

```
solar-weather-assistant/
├── app.py              <- interface Gradio
├── src/
│   ├── ingest.py       <- indexa docs/ no ChromaDB
│   └── rag.py          <- chain RAG (LangChain + Groq)
├── docs/               <- base de conhecimento (Markdown)
├── .chroma/            <- indice vetorial (gerado pelo ingest.py)
├── requirements.txt
├── render.yaml         <- configuracao de deploy (Render)
└── setup.py            <- script de configuracao interativo
```

## Projeto relacionado

Este assistente usa como base de conhecimento os artefatos do projeto de IC:
[cme-flare-prediction](https://github.com/gabriel-aleksandro/cme-flare-prediction) —
predicao de flares solares e CMEs com biLSTM + Atencao (UNICAMP/PIBIC CNPq).

O modelo de previsao esta publicado no Hugging Face:
[gabriel-aleksandro/solar-flare-bilstm](https://huggingface.co/gabriel-aleksandro/solar-flare-bilstm)

## Licenca

MIT
