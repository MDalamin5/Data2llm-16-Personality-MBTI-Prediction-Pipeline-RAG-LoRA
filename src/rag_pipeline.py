import os
import re
import faiss
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

from prompts import rag_prompt
from schemas import UserNameExtract
from dotenv import load_dotenv
load_dotenv()

def initialize_models():
    """Initializes the language model and embeddings."""
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm, embeddings

def create_vector_store(path="processed/", glob="**/*.txt"):
    """Loads documents, adds metadata, and creates a FAISS vector store."""
    def extract_name_from_text(text):
        match = re.search(r"Name:\s*(.+)", text)
        return match.group(1).strip() if match else "Unknown"

    loader = DirectoryLoader(
        path=path,
        glob=glob,
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

    for doc in documents:
        name = extract_name_from_text(doc.page_content)
        doc.metadata["name"] = name

    llm, embeddings = initialize_models()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_rag_chain(vector_store, user_query):
    """Creates the RAG chain with a retriever filtered by the extracted name."""
    llm, _ = initialize_models()
    llm_with_str_output = llm.with_structured_output(UserNameExtract)
    time.sleep(2)
    result = llm_with_str_output.invoke(user_query)
    search_name = result.user_name

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 1, "filter": {"name": search_name}}
    )

    rag_prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=rag_prompt
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain