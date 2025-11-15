


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re

app = FastAPI(title="MBTI Personality API - Lightning Studio")

# Enable CORS for local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------
# MBTI Model Setup (loaded once at startup)
# --------------------------------------------------------------
print("🔄 Loading MBTI model...")

model_name = "microsoft/Phi-3-mini-4k-instruct"
adapter_name = "alam1n/phi3-mbti-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, adapter_name)

tokenizer = AutoTokenizer.from_pretrained(adapter_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ MBTI model loaded successfully!")

# --------------------------------------------------------------
# MBTI Table (16 types)
# --------------------------------------------------------------
MBTI_TABLE = {
    "INTJ": ("Strategic, independent", "Original minds with drive for goals", "Strong in planning roles, e.g., tech startups"),
    "INTP": ("Analytical, inventive", "Seek logical explanations", "Great for R&D in Dhaka AI labs"),
    "ENTJ": ("Decisive, leader", "Assume leadership readily", "Ideal for scaling startups"),
    "ENTP": ("Innovative, debating", "Resourceful in new challenges", "Perfect for product ideation"),
    "INFJ": ("Insightful, idealistic", "Seek meaning & connection", "Excellent in HR / culture building"),
    "INFP": ("Creative, empathetic", "Loyal to values", "Strong in content & community"),
    "ENFJ": ("Charismatic, mentoring", "Attuned to others' emotions", "Great for sales & partnerships"),
    "ENFP": ("Enthusiastic, imaginative", "See possibilities", "Best for marketing & outreach"),
    "ISTJ": ("Dependable, thorough", "Responsible & detail-oriented", "Core for operations & compliance"),
    "ISFJ": ("Conscientious, supportive", "Committed to obligations", "Perfect for customer success"),
    "ESTJ": ("Organized, decisive", "Implement decisions fast", "Strong in project management"),
    "ESFJ": ("Harmonious, cooperative", "Want harmony", "Ideal for team coordination"),
    "ISTP": ("Practical, adaptable", "Quick problem-solver", "Excellent in DevOps / troubleshooting"),
    "ISFP": ("Artistic, flexible", "Enjoy present moment", "Great for UI/UX design"),
    "ESTP": ("Action-oriented, pragmatic", "Focus on immediate results", "Perfect for field sales"),
    "ESFP": ("Outgoing, spontaneous", "Enjoy working with people", "Best for events & community"),
}

# --------------------------------------------------------------
# Request Model - Simplified for RAG output
# --------------------------------------------------------------
class PersonInfo(BaseModel):
    text: str  # Main field - accepts any text from RAG
    name: str = None
    headline: str = None
    location: str = None
    about: str = None
    posts: str = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Al Amin is a Senior Software Engineer based in Dhaka, passionate about AI and mentoring juniors."
            }
        }

# --------------------------------------------------------------
# MBTI Prediction Function
# --------------------------------------------------------------
def predict_mbti(person_text: str) -> dict:
    # Truncate if too long
    if len(person_text) > 3200:
        person_text = person_text[:3200] + "..."

    prompt = f"""<|system|>
You are an MBTI expert. Return ONLY the 4-letter type.<|end|>
<|user|>
Analyze:

"{person_text}"<|end|>
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        try:
            # Try with cache disabled first
            output = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        except Exception as e:
            # Fallback: try with minimal parameters
            print(f"Generation with use_cache=False failed: {e}")
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id,
            )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract MBTI using regex
    match = re.findall(r'\b[IE][NS][FT][JP]\b', reply.upper())
    mbti = match[-1] if match else "UNKNOWN"

    # Get details from table
    traits, desc, context = MBTI_TABLE.get(mbti, ("–", "–", "–"))

    return {
        "mbti_type": mbti,
        "key_traits": traits,
        "description": desc,
        "business_fit": context,
        "raw_output": reply
    }

# --------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "MBTI Personality API - Lightning Studio",
        "status": "online",
        "version": "1.0",
        "model": "Phi-3 + LoRA MBTI",
        "endpoints": {
            "predict": "/api/predict (POST)",
            "health": "/health",
            "info": "/api/info"
        }
    }

@app.get("/api/info")
def info():
    hostname = socket.gethostname()
    return {
        "hostname": hostname,
        "message": "API is running",
        "model_loaded": True,
        "endpoints": [
            "/",
            "/api/info",
            "/api/predict",
            "/api/test",
            "/health"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else "N/A"
    }

@app.get("/api/test")
def test():
    return {
        "status": "working",
        "data": "API is live and accessible!",
        "source": "Lightning AI Studio"
    }

@app.post("/api/data")
def process_data(payload: dict):
    return {
        "received": payload,
        "processed": True,
        "message": "Data processed successfully"
    }

@app.post("/api/predict")
def predict_personality(person: PersonInfo):
    """
    Predict MBTI personality type from person information.
    
    Send either structured fields (name, headline, about, etc.) 
    or just a 'text' field with all information.
    """
    try:
        # Build person text from fields or use raw text
        if person.text:
            person_text = person.text
        else:
            parts = []
            if person.name:
                parts.append(f"Name: {person.name}")
            if person.headline:
                parts.append(f"Headline: {person.headline}")
            if person.location:
                parts.append(f"Location: {person.location}")
            if person.about:
                parts.append(f"About: {person.about}")
            if person.posts:
                parts.append(f"Posts: {person.posts}")
            
            person_text = "\n".join(parts)
        
        if not person_text.strip():
            raise HTTPException(status_code=400, detail="No text provided for analysis")
        
        # Get prediction
        result = predict_mbti(person_text)
        
        # Add input info
        result["input_length"] = len(person_text)
        result["success"] = True
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --------------------------------------------------------------
# Startup
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MBTI Personality API Starting...")
    print("="*60)
    print("📍 Local URL: http://0.0.0.0:8000")
    print("🌍 Public URL: Check Lightning Studio or use tunnel")
    print("🤖 Model: Phi-3 + MBTI LoRA")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )