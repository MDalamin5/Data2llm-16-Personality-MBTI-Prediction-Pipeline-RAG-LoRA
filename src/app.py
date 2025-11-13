# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import create_vector_store, create_rag_chain, initialize_models
import nest_asyncio
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import threading
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
# === CRITICAL FIX: Apply nest_asyncio ONLY in thread-local context ===
# We'll apply it safely inside the request handler
def safe_apply_nest_asyncio():
    if not hasattr(safe_apply_nest_asyncio, "applied"):
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            if "uvloop" not in str(type(loop)):
                nest_asyncio.apply()
        except RuntimeError:
            # No running loop yet
            pass
        safe_apply_nest_asyncio.applied = True

# Initialize shared components (outside request to avoid re-creating)
print("Initializing models and vector store...")

llm, _ = initialize_models()
vector_store = create_vector_store()
config = RailsConfig.from_path("config")
guard_rail = RunnableRails(config=config, llm=llm)
print("Initialization complete.")

app = FastAPI(
    title="RAG with Guardrails API",
    description="API for querying the RAG system with guardrails for personality prediction.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def handle_query(request: QueryRequest):
    try:
        # Apply nest_asyncio safely inside request if needed
        safe_apply_nest_asyncio()

        # Create RAG chain per request (name extraction depends on query)
        rag_chain = create_rag_chain(vector_store, request.query)
        guard_with_rag_chain = guard_rail | rag_chain

        # Run the chain
        result = guard_with_rag_chain.invoke(request.query)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")