from rag_pipeline import create_vector_store, create_rag_chain

if __name__ == "__main__":
    # Create the vector store from the documents
    vector_store = create_vector_store()

    # Get user input
    user_query = input("Enter your query: ")

    # Create and run the RAG chain
    rag_chain = create_rag_chain(vector_store, user_query)
    result = rag_chain.invoke(user_query)

    # Print the result
    print(result)