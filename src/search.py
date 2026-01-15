import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


def get_embeddings():
    """Retorna o modelo de embeddings baseado no provider configurado"""
    provider = EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)
    
    else:
        raise RuntimeError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def get_llm():
    """Retorna o modelo de LLM baseado no provider configurado"""
    provider = LLM_PROVIDER.lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
        return ChatOpenAI(model=model, temperature=0)
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")


def get_vector_store():
    """Retorna a conexão com o banco vetorial"""
    for k in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")
    
    embeddings = get_embeddings()
    
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    
    return store


def format_docs(docs):
    """Formata os documentos recuperados para o contexto"""
    return "\n\n".join([doc.page_content for doc in docs])


def search_prompt(question=None):
    """
    Cria a chain de busca e resposta.
    Se question for None, retorna a chain para uso interativo.
    Se question for fornecida, executa a busca e retorna a resposta.
    """
    try:
        # Inicializa componentes
        vector_store = get_vector_store()
        llm = get_llm()
        
        # Cria o retriever com k=10
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Cria o prompt template
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Cria a chain
        chain = (
            {
                "contexto": retriever | format_docs,
                "pergunta": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Se uma pergunta foi fornecida, executa e retorna a resposta
        if question:
            return chain.invoke(question)
        
        # Caso contrário, retorna a chain para uso posterior
        return chain
        
    except Exception as e:
        print(f"Erro ao inicializar o sistema de busca: {e}")
        return None


def search_with_details(question, show_sources=False):
    """
    Busca e retorna a resposta com opção de mostrar as fontes.
    """
    try:
        vector_store = get_vector_store()
        llm = get_llm()
        
        # Busca os documentos mais relevantes
        results = vector_store.similarity_search_with_score(question, k=10)
        
        # Formata o contexto
        contexto = "\n\n".join([doc.page_content for doc, score in results])
        
        # Monta o prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_value = prompt_template.format(contexto=contexto, pergunta=question)
        
        # Chama a LLM
        response = llm.invoke(prompt_value)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        if show_sources:
            return {
                "answer": answer,
                "sources": results
            }
        
        return answer
        
    except Exception as e:
        return f"Erro ao processar a pergunta: {e}"