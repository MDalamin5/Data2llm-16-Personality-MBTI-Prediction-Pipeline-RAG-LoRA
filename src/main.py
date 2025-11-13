from rag_pipeline import create_vector_store, create_rag_chain, initialize_models
import nest_asyncio
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

if __name__ == "__main__":
    nest_asyncio.apply()
    llm, _ = initialize_models()
    # Create the vector store from the documents
    vector_store = create_vector_store()

    # Get user input
    user_query = input("Enter your query: ")

    # Create and run the RAG chain
    rag_chain = create_rag_chain(vector_store, user_query)
    config = RailsConfig.from_path("config")
    guard_rail = RunnableRails(config=config, llm=llm)
    guard_with_rag_chain = guard_rail | rag_chain

    result = guard_with_rag_chain.invoke(user_query)

    # Print the result
    print(result)