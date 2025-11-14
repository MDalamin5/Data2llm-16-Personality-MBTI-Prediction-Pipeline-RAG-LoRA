# # app.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from rag_pipeline import create_vector_store, create_rag_chain, initialize_models
# import nest_asyncio
# from nemoguardrails import RailsConfig
# from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
# import threading
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
# # === CRITICAL FIX: Apply nest_asyncio ONLY in thread-local context ===
# # We'll apply it safely inside the request handler
# def safe_apply_nest_asyncio():
#     if not hasattr(safe_apply_nest_asyncio, "applied"):
#         try:
#             import asyncio
#             loop = asyncio.get_running_loop()
#             if "uvloop" not in str(type(loop)):
#                 nest_asyncio.apply()
#         except RuntimeError:
#             # No running loop yet
#             pass
#         safe_apply_nest_asyncio.applied = True

# # Initialize shared components (outside request to avoid re-creating)
# print("Initializing models and vector store...")

# llm, _ = initialize_models()
# vector_store = create_vector_store()
# config = RailsConfig.from_path("config")
# guard_rail = RunnableRails(config=config, llm=llm)
# print("Initialization complete.")

# app = FastAPI(
#     title="RAG with Guardrails API",
#     description="API for querying the RAG system with guardrails for personality prediction.",
#     version="1.0.0"
# )

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/query")
# def handle_query(request: QueryRequest):
#     try:
#         # Apply nest_asyncio safely inside request if needed
#         safe_apply_nest_asyncio()

#         # Create RAG chain per request (name extraction depends on query)
#         rag_chain = create_rag_chain(vector_store, request.query)
#         guard_with_rag_chain = guard_rail | rag_chain

#         # Run the chain
#         result = guard_with_rag_chain.invoke(request.query)
#         return {"result": result}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# app.py
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import create_vector_store, create_rag_chain, initialize_models
import nest_asyncio
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === CRITICAL FIX: Apply nest_asyncio ONLY in thread-local context ===
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
    title="RAG with Guardrails + MBTI Prediction API",
    description="API for querying the RAG system with guardrails and personality prediction.",
    version="1.0.0"
)

# Configuration for Lightning AI endpoint
LIGHTNING_API_URL = os.getenv(
    "LIGHTNING_API_URL", 
    "https://kind-cloths-smash.loca.lt/api/predict"
)

class QueryRequest(BaseModel):
    query: str

class QueryWithPredictionRequest(BaseModel):
    query: str
    predict_personality: bool = True  # Enable/disable prediction

def call_mbti_prediction(text: str) -> dict:
    """
    Call the Lightning AI MBTI prediction endpoint.
    Returns prediction result or error info.
    """
    try:
        # Clean the text
        cleaned_text = str(text).strip()
        
        if not cleaned_text:
            return {
                "success": False,
                "error": "Empty text provided for prediction"
            }
        
        # Prepare payload
        payload = {"text": cleaned_text}
        
        print(f"📤 Sending to Lightning API: {LIGHTNING_API_URL}")
        print(f"📝 Text length: {len(cleaned_text)} chars")
        
        response = requests.post(
            LIGHTNING_API_URL,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            return {
                "success": True,
                "prediction": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}",
                "details": response.text[:200]  # Limit error details
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Prediction API timeout (30s exceeded)"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to connect to prediction API: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

@app.get("/")
def root():
    return {
        "message": "RAG with Guardrails + MBTI Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query (POST) - RAG query only",
            "query_with_prediction": "/query-with-prediction (POST) - RAG + MBTI prediction",
            "health": "/health (GET)",
            "config": "/config (GET)"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rag_initialized": vector_store is not None,
        "guardrails_initialized": guard_rail is not None,
        "prediction_endpoint": LIGHTNING_API_URL
    }

@app.get("/config")
def get_config():
    return {
        "lightning_api_url": LIGHTNING_API_URL,
        "prediction_enabled": True
    }

@app.post("/query")
def handle_query(request: QueryRequest):
    """
    Original endpoint - RAG query only, no personality prediction.
    """
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

@app.post("/query-with-prediction")
def handle_query_with_prediction(request: QueryWithPredictionRequest):
    """
    Enhanced endpoint - RAG query + MBTI personality prediction.
    
    The RAG result is sent to Lightning AI for personality analysis.
    """
    try:
        # Apply nest_asyncio safely inside request if needed
        safe_apply_nest_asyncio()

        # Create RAG chain per request
        rag_chain = create_rag_chain(vector_store, request.query)
        guard_with_rag_chain = guard_rail | rag_chain

        # Run the chain
        rag_result = guard_with_rag_chain.invoke(request.query)
        # rag_result = "Al Amin is a Senior Software Engineer based in Dhaka, passionate about AI and mentoring juniors."
        
        response = {
            "query": request.query,
            "rag_result": rag_result,
            "personality_prediction": None
        }

        # If prediction is enabled, call Lightning AI endpoint
        if request.predict_personality:
            # Format the RAG result as needed (currently just converting to string)
            # You can add custom formatting here if needed
            formatted_text = str(rag_result).strip()
            
            # Optional: Add more context or formatting
            # formatted_text = f"Person Information:\n{formatted_text}"
            
            prediction_result = call_mbti_prediction(formatted_text)
            response["personality_prediction"] = prediction_result

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query with prediction: {str(e)}"
        )