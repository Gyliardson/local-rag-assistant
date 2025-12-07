import streamlit as st
import tempfile
import os
import time
from typing import Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever

# --- CONSTANTS ---
MODEL_EMBEDDING = "bge-m3"
MODEL_LLM = "llama3.1.3"  # Configurável pelo usuário no código
PERSIST_DIRECTORY = "./vectorstore"

st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖", layout="wide")

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    try:
        with st.status("Processando documento...", expanded=True) as status:
            st.write("📖 Lendo arquivo PDF...")
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            st.write("✂️ Segmentando texto...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            total_chunks = len(splits)
            
            st.write(f"💾 Indexando {total_chunks} chunks...")
            progress_bar = st.progress(0)
            
            embedding_function = OllamaEmbeddings(model=MODEL_EMBEDDING)
            
            # Inicializa conexão com o banco
            vectorstore = Chroma(
                embedding_function=embedding_function,
                persist_directory=PERSIST_DIRECTORY
            )
            
            # Inserção em lotes (Batch) para atualizar a barra de progresso
            BATCH_SIZE = 20
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = splits[i : i + BATCH_SIZE]
                vectorstore.add_documents(batch)
                
                # Atualiza UI
                progress = min((i + BATCH_SIZE) / total_chunks, 1.0)
                progress_bar.progress(progress)

            status.update(label="Concluído!", state="complete", expanded=False)
            
            return vectorstore.as_retriever(search_kwargs={"k": 3})
            
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        raise e
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def format_docs(docs) -> str:
    """Combina o conteúdo dos documentos recuperados em uma única string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.title("🤖 RAG Local: Chat com Documentos")
    st.markdown("""
    Este assistente utiliza **LangChain**, **Ollama** e **ChromaDB** para permitir conversas 
    contextuais com seus arquivos PDF, rodando 100% localmente para privacidade e controle de dados.
    """)

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Configuração")
        uploaded_file = st.file_uploader("Carregue seu PDF", type="pdf")
        
        st.divider()
        st.markdown("**Stack Técnica:**")
        st.markdown("- Python & Streamlit")
        st.markdown(f"- LLM: {MODEL_LLM}")
        st.markdown(f"- Embeddings: {MODEL_EMBEDDING}")
        
        if st.button("🗑️ Limpar Conversa"):
            st.session_state.messages = []
            st.rerun()

    # --- APP LOGIC ---
    if uploaded_file:
        # Gerenciamento de Estado para evitar reprocessamento desnecessário
        current_file_id = uploaded_file.file_id
        if "retriever" not in st.session_state or st.session_state.get("file_id") != current_file_id:
            st.session_state.retriever = process_pdf(uploaded_file)
            st.session_state.file_id = current_file_id
            st.session_state.messages = [] # Reset chat on new file
        
        retriever = st.session_state.retriever

        # Configuração da Chain (LCEL Architecture)
        llm = ChatOllama(model=MODEL_LLM, temperature=0.1)
        
        prompt_template = """
        Você é um assistente técnico de análise documental. 
        Use estritamente o contexto abaixo para responder à pergunta do usuário.

        Diretrizes:
        1. Responda de forma clara e direta.
        2. Se a informação não estiver no contexto, NÃO invente. Em vez disso, responda educadamente:
           "Analisei o documento fornecido e não encontrei informações específicas sobre esse tema."
        
        Contexto:
        {context}

        Pergunta: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Interface de Chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if input_text := st.chat_input("Digite sua pergunta sobre o documento..."):
            with st.chat_message("user"):
                st.markdown(input_text)
            st.session_state.messages.append({"role": "user", "content": input_text})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                start_time = time.time()
                
                try:
                    for chunk in rag_chain.stream(input_text):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    elapsed = time.time() - start_time
                    st.caption(f"⚡ Latência: {elapsed:.2f}s | Modelo: {MODEL_LLM}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Erro na geração: {e}")

    else:
        st.info("👈 Faça o upload de um PDF na barra lateral para iniciar o sistema.")

if __name__ == "__main__":
    main()