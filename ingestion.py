import os
import sys
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONSTANTS ---
DOC_PATH = "./data/manual.pdf"  # Caminho padrão (pode ser alterado ou receber via arg)
DB_PATH = "./vectorstore"
MODEL_EMBEDDING = "bge-m3"

def load_and_split_document(file_path: str) -> List:
    """
    Carrega um documento PDF e o divide em chunks menores para processamento.
    
    Args:
        file_path (str): Caminho absoluto ou relativo para o arquivo PDF.
        
    Returns:
        List: Lista de documentos segmentados (chunks).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo '{file_path}' não foi encontrado.")

    print(f"Carregando: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print(f"✂️  Segmentando {len(docs)} páginas...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(docs)

def ingest_to_vectorstore(splits: List, db_path: str, model_name: str) -> None:
    """
    Gera embeddings para os chunks e os persiste no banco vetorial ChromaDB.
    
    Args:
        splits (List): Lista de chunks de texto.
        db_path (str): Diretório de persistência do banco.
        model_name (str): Nome do modelo de embedding do Ollama.
    """
    print(f"Inicializando Embedding ({model_name})...")
    embedding_function = OllamaEmbeddings(model=model_name)

    print(f"Inserindo {len(splits)} vetores no ChromaDB em '{db_path}'...")
    
    # Criação do banco com persistência automática
    Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=db_path
    )
    print("Ingestão concluída com sucesso!")

def main():
    try:
        # Pipeline de Execução
        print("--- Início do Processo de Ingestão (Batch) ---")
        
        splits = load_and_split_document(DOC_PATH)
        ingest_to_vectorstore(splits, DB_PATH, MODEL_EMBEDDING)
        
    except FileNotFoundError as e:
        print(f"❌ Erro de Arquivo: {e}")
    except Exception as e:
        print(f"❌ Erro Inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()