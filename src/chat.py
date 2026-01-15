import os
from search import search_prompt, search_with_details

def print_welcome():
    """Imprime mensagem de boas-vindas"""
    print("=" * 60)
    print("🤖 Chat RAG - Sistema de Consulta")
    print("=" * 60)
    
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    
    print(f"\n📊 Embeddings: {embedding_provider.upper()}")
    print(f"🧠 LLM: {llm_provider.upper()}")
    print("\n💡 Comandos disponíveis:")
    print("  - Digite sua pergunta para buscar no documento")
    print("  - Digite 'sources' para ver as fontes da última resposta")
    print("  - Digite 'clear' para limpar a tela")
    print("  - Digite 'sair' ou 'exit' para encerrar")
    print("=" * 60)
    print()


def print_sources(sources):
    """Imprime as fontes dos documentos"""
    print("\n" + "=" * 60)
    print("📚 FONTES CONSULTADAS")
    print("=" * 60)
    
    for i, (doc, score) in enumerate(sources, start=1):
        print(f"\n--- Fonte {i} (relevância: {score:.4f}) ---")
        print(f"\n{doc.page_content[:300]}...")
        
        if doc.metadata:
            print("\nMetadados:")
            for k, v in doc.metadata.items():
                print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)


def main():
    """Função principal do chat"""
    print_welcome()
    
    # Inicializa a chain
    chain = search_prompt()
    
    if not chain:
        print("❌ Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    print("✅ Sistema inicializado com sucesso!\n")
    
    last_question = None
    
    # Loop principal do chat
    while True:
        try:
            # Recebe a pergunta do usuário
            question = input("🧑 Você: ").strip()
            
            # Comandos especiais
            if question.lower() in ['sair', 'exit', 'quit']:
                print("\n👋 Encerrando o chat. Até logo!")
                break
            
            if question.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print_welcome()
                continue
            
            if question.lower() == 'sources':
                if last_question:
                    print("\n🔍 Buscando fontes...")
                    result = search_with_details(last_question, show_sources=True)
                    print_sources(result['sources'])
                else:
                    print("\n⚠️  Nenhuma pergunta foi feita ainda.")
                continue
            
            if not question:
                continue
            
            # Processa a pergunta
            print("\n🤖 Assistente: ", end="", flush=True)
            
            # Invoca a chain e obtém a resposta
            answer = chain.invoke(question)
            print(answer)
            print()
            
            # Salva a última pergunta para o comando 'sources'
            last_question = question
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrompido. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")


if __name__ == "__main__":
    main()