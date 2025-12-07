# Local RAG Assistant

> **Chat inteligente com documentos PDF, rodando 100% localmente.**

## Sobre o Projeto

Este projeto é um assistente de **RAG (Retrieval-Augmented Generation)** desenvolvido para permitir a análise de documentos privados com total privacidade. Diferente de ferramentas baseadas em nuvem, aqui seus dados nunca saem da sua máquina.

O sistema ingere documentos PDF, processa-os em vetores matemáticos e utiliza um Grande Modelo de Linguagem (LLM) para responder perguntas complexas com base no contexto, citando fontes e evitando alucinações.

## Demonstração

[Clique aqui para assistir ao vídeo de demonstração](https://youtu.be/Lltl1Js1pmM)

## Funcionalidades Principais

* **100% Offline:** Privacidade total utilizando Ollama para inferência e ChromaDB para armazenamento vetorial.
* **Ingestão em Lote (Batch):** Implementação de barra de progresso visual para processamento de arquivos grandes.
* **Suporte Multilíngue:** Capacidade de ler documentos em inglês e responder perguntas fluentemente em português.
* **Segurança na Resposta:** Calibrado para admitir quando não sabe a resposta, evitando inventar informações.

## Stack Tecnológica

* **Frontend:** Streamlit
* **Orquestração:** LangChain (LCEL)
* **Banco Vetorial:** ChromaDB
* **Engine de IA:** Ollama
* **Modelos:**
    * LLM: `llama3.1` (8B)
    * Embeddings: `bge-m3`

## Pré-requisitos

É necessário ter o **Ollama** instalado e rodando.

1.  **Instale o Ollama:** [ollama.com](https://ollama.com).
2.  **Baixe os modelos:**

~~~bash
ollama pull llama3.1
ollama pull bge-m3
~~~

> **Nota sobre Hardware:**
> O projeto foi configurado para o modelo **llama3.1 (8B)**.
> Para computadores com pouca RAM ou sem GPU, utilize um modelo mais leve:
> 1. Baixe: `ollama pull llama3.2:3b`
> 2. Nos arquivos `app.py` e `ingestion.py`, altere `MODEL_LLM` para `"llama3.2:3b"`.

## Instalação e Execução

1.  **Clone o repositório:**
    ~~~bash
    git clone https://github.com/Gyliardson/local-rag-assistant.git
    cd local-rag-assistant
    ~~~

2.  **Ambiente virtual:**

    *Windows:*
    ~~~bash
    python -m venv venv
    venv\Scripts\activate
    ~~~

    *Linux/Mac:*
    ~~~bash
    python3 -m venv venv
    source venv/bin/activate
    ~~~

3.  **Dependências:**
    ~~~bash
    pip install -r requirements.txt
    ~~~

4.  **Execução:**
    ~~~bash
    streamlit run app.py
    ~~~

O navegador abrirá automaticamente em [http://localhost:8501](http://localhost:8501).

## Performance

Validado em GPU **Intel Arc B580**. Compatível com:
* NVIDIA (CUDA)
* AMD (ROCm)
* Apple Silicon (M1/M2/M3)
* CPU (Maior latência)

## Contato

Sinta-se à vontade para contribuir com o projeto.

* **Desenvolvido por:** Gyliardson Keitison
* **LinkedIn:** [Gyliardson keitison](https://www.linkedin.com/in/gyliardson-keitison/)

---
*Este projeto faz parte do meu portfólio de transição de carreira para Engenharia de IA.*
