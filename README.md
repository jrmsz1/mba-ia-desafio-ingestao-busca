# Sistema RAG (Retrieval-Augmented Generation) com pgVector

Sistema completo de RAG que realiza a ingestão de documentos PDF, converte o conteúdo em embeddings, armazena no PostgreSQL com pgVector e disponibiliza um chat interativo via CLI para consultas semânticas.

## Requisitos

- Python 3.8+
- PostgreSQL com extensão pgVector instalada
- Conta OpenAI com API Key ativa **OU** Google AI API Key ativa

## Instalação

### 1. Configurar ambiente virtual Python

Crie e ative um ambiente virtual antes de instalar as dependências:

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependências Python

Com o ambiente virtual ativado, instale as dependências:

```bash
pip install python-dotenv langchain-community langchain-text-splitters langchain-openai langchain-google-genai langchain-core langchain-postgres pypdf psycopg
```

Ou instale com:

```bash
pip install -r requirements.txt
```

### 3. Configurar PostgreSQL com pgVector

Certifique-se de que o PostgreSQL está rodando e a extensão pgVector está instalada.

#### Usando Docker Compose (Recomendado):

Suba o banco de dados:

```bash
docker compose up -d
```

#### Ou usando Docker (comando direto):

```bash
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

#### Ou instalando manualmente:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Configuração

### 1. Criar arquivo `.env`

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```dotenv
# Provider Configuration
EMBEDDING_PROVIDER=openai  # Opções: "openai" ou "google"
LLM_PROVIDER=openai        # Opções: "openai" ou "google"

# Google AI Configuration
GOOGLE_API_KEY=sua-chave-google-aqui
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_LLM_MODEL=gemini-2.5-flash-lite

# OpenAI Configuration
OPENAI_API_KEY=sua-chave-openai-aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-5-nano

# Database Configuration
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# pgVector Configuration
PG_VECTOR_COLLECTION_NAME=rag

# PDF Configuration
PDF_PATH=document.pdf
```

**Nota sobre Providers:**
- `EMBEDDING_PROVIDER`: Define qual provedor usar para gerar embeddings (vetorização)
- `LLM_PROVIDER`: Define qual LLM usar para gerar respostas no chat
- Você pode usar OpenAI para embeddings e Google para LLM (ou vice-versa)
- Preencha apenas as API Keys dos providers que for utilizar

### 2. Adicionar seu documento PDF

Coloque o arquivo PDF que deseja processar no diretório do projeto e atualize a variável `PDF_PATH` no arquivo `.env` com o caminho correto.

## Estrutura do Projeto

```
.
├── src/
│   ├── ingest.py          # Script de ingestão de PDFs
│   ├── search.py          # Módulo de busca vetorial e LLM
│   └── chat.py            # Interface CLI do chat interativo
├── docker-compose.yml     # Configuração do PostgreSQL
├── .env                   # Variáveis de ambiente (não commitar!)
├── document.pdf           # Seu documento PDF
├── requirements.txt       # Dependências Python
├── venv/                  # Ambiente virtual Python (não commitar!)
└── README.md             # Este arquivo
```

## Execução

### Ordem de Execução

Siga estes passos na ordem correta:

#### 1. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde alguns segundos para o PostgreSQL inicializar completamente.

#### 2. Executar a ingestão do PDF

Processe e armazene o documento no banco vetorial:

```bash
python src/ingest.py
```

**Saída esperada:**

```
Loading PDF: document.pdf
Splitting documents into chunks...
Created 45 chunks
Creating embeddings...
Using OpenAI embeddings: text-embedding-3-small
Connecting to PostgreSQL with pgVector...
Storing documents in database...
Successfully ingested 45 chunks into the database
```

#### 3. Iniciar o chat interativo

Após a ingestão bem-sucedida, inicie o chat para fazer perguntas:

```bash
python src/chat.py
```

**Interface do chat:**

```
============================================================
🤖 Chat RAG - Sistema de Consulta
============================================================

📊 Embeddings: OPENAI
🧠 LLM: OPENAI

💡 Comandos disponíveis:
  - Digite sua pergunta para buscar no documento
  - Digite 'sources' para ver as fontes da última resposta
  - Digite 'clear' para limpar a tela
  - Digite 'sair' ou 'exit' para encerrar
============================================================

✅ Sistema inicializado com sucesso!

🧑 Você: Qual é o tema principal do documento?
🤖 Assistente: [Resposta baseada no conteúdo do PDF]

🧑 Você: sources
🔍 Buscando fontes...
============================================================
📚 FONTES CONSULTADAS
============================================================
[Mostra os 10 trechos mais relevantes consultados]
```

## Como Funciona

### Ingestão (ingest.py)

1. **Carregamento do PDF**: O documento é lido usando `PyPDFLoader`
2. **Divisão em chunks**: O texto é dividido em pedaços de 1000 caracteres com overlap de 150 caracteres
3. **Geração de embeddings**: Cada chunk é convertido em vetor usando o modelo configurado (OpenAI ou Google)
4. **Armazenamento**: Os vetores são salvos no PostgreSQL com pgVector para busca semântica

### Consulta (search.py + chat.py)

1. **Vetorização da pergunta**: A pergunta do usuário é convertida em embedding
2. **Busca semântica**: Busca os 10 chunks mais relevantes (k=10) no banco vetorial usando similaridade de cosseno
3. **Montagem do prompt**: Monta um prompt estruturado com o contexto recuperado e a pergunta
4. **Chamada à LLM**: Envia o prompt para a LLM configurada (OpenAI ou Google)
5. **Resposta**: Retorna a resposta gerada baseada apenas no contexto fornecido

### Regras do Sistema

O sistema segue regras rigorosas para evitar alucinações:
- ✅ Responde **apenas** com base no contexto recuperado
- ❌ **Nunca** inventa informações
- ❌ **Nunca** usa conhecimento externo ao documento
- ℹ️ Se a informação não estiver no contexto, informa explicitamente ao usuário

## Parâmetros Configuráveis

### Chunking (ingest.py)
- **chunk_size**: 1000 caracteres
- **chunk_overlap**: 150 caracteres
- **add_start_index**: False

### Busca Vetorial (search.py)
- **k**: 10 documentos mais relevantes
- **search_type**: similarity (busca por similaridade de cosseno)

### LLM
- **temperature**: 0 (respostas determinísticas)

Esses parâmetros podem ser ajustados diretamente nos arquivos conforme necessário.

## Solução de Problemas

### Erro: "Environment variable X is not set"

Verifique se todas as variáveis necessárias estão configuradas no arquivo `.env`.

### Erro: "PDF file not found"

Certifique-se de que o caminho do PDF no `.env` está correto e o arquivo existe.

### Erro de conexão com PostgreSQL

- Verifique se o PostgreSQL está rodando
- Confirme as credenciais e porta no `DATABASE_URL`
- Se usar Docker, use `host.docker.internal` ao invés de `localhost` quando o script roda dentro de um container

### Erro com API OpenAI

- Verifique se sua chave API está ativa
- Confirme se há créditos disponíveis na conta OpenAI
- Verifique se o modelo está correto (`gpt-5-nano` para LLM, `text-embedding-3-small` para embeddings)

### Erro com API Google

- Verifique se sua chave API está ativa no Google AI Studio
- Confirme se os modelos estão corretos (`gemini-2.5-flash-lite` para LLM, `models/embedding-001` para embeddings)

### Chat não encontra respostas no documento

- Verifique se a ingestão foi concluída com sucesso
- Confirme que o `EMBEDDING_PROVIDER` é o mesmo na ingestão e no chat
- Tente reformular a pergunta de forma mais específica
- Use o comando `sources` para ver quais trechos foram consultados

## Funcionalidades

### Ingestão de Documentos
- ✅ Suporte para arquivos PDF
- ✅ Chunking inteligente com overlap
- ✅ Suporte para embeddings OpenAI e Google
- ✅ Armazenamento vetorial com pgVector
- ✅ Preservação de metadados do documento

### Chat Interativo
- ✅ Interface CLI amigável
- ✅ Busca semântica com 10 resultados mais relevantes
- ✅ Respostas baseadas apenas no contexto do documento
- ✅ Comando para visualizar fontes consultadas
- ✅ Suporte para múltiplos providers de LLM
- ✅ Prevenção de alucinações com prompt estruturado
- ✅ Comandos úteis (sources, clear, exit)

## Comandos do Chat

| Comando | Descrição |
|---------|-----------|
| `<sua pergunta>` | Faz uma pergunta sobre o documento |
| `sources` | Mostra as fontes da última resposta |
| `clear` | Limpa a tela do terminal |
| `sair` / `exit` | Encerra o chat |

## Exemplos de Uso

### Exemplo 1: Pergunta dentro do contexto

```
🧑 Você: Quais são os principais tópicos abordados no documento?
🤖 Assistente: Com base no documento, os principais tópicos são...
```

### Exemplo 2: Pergunta fora do contexto

```
🧑 Você: Qual é a capital da França?
🤖 Assistente: Não tenho informações necessárias para responder sua pergunta.
```

### Exemplo 3: Consultando fontes

```
🧑 Você: sources
🔍 Buscando fontes...
============================================================
📚 FONTES CONSULTADAS
============================================================

--- Fonte 1 (relevância: 0.8542) ---
[Trecho do documento que foi usado como contexto]
...
```

## Próximos Passos

Após configurar o sistema, você pode:

- Adicionar mais documentos ao banco vetorial
- Implementar uma API REST para consultas
- Criar interface web com Streamlit ou Gradio
- Adicionar suporte para outros formatos (DOCX, TXT, etc.)
- Implementar cache de respostas para perguntas frequentes
- Adicionar filtros por metadados na busca
- Integrar com sistemas de autenticação

## Arquitetura

```
┌──────────────┐
│  document.pdf│
└───────┬──────┘
       │
       ▼
┌─────────────────┐
│   ingest.py     │ ─── Chunking (1000 chars, overlap 150)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │ ─── OpenAI ou Google
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PostgreSQL +   │ ─── Armazenamento vetorial
│    pgVector     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    search.py    │ ─── Busca semântica (k=10)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     LLM         │ ─── OpenAI ou Google
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    chat.py      │ ─── Interface CLI
└─────────────────┘
```

## Licença

Este projeto é fornecido como exemplo educacional.