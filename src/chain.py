import os
import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# Cargar modelo una sola vez al iniciar
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model = PeftModel.from_pretrained(base_model, "Razo3000/mtg-oracle")
model.eval()
print("✓ Model loaded!")

def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name="mtg_rulings", embedding_function=ef)

def query_model(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

def ask(question: str) -> str:
    collection = get_collection()
    
    # 1. Buscar rulings relevantes en ChromaDB
    results = collection.query(query_texts=[question], n_results=5)
    context = "\n\n".join(results["documents"][0])
    
    # 2. Construir prompt
    prompt = f"""System: You are MTG Oracle, an expert Magic: The Gathering judge and analyst.
You explain rules interactions clearly and evaluate cards with authority.
Respond based ONLY on the rulings provided below.
If the information is not in the rulings, say so clearly.

Relevant rulings:
{context}

User: {question}
Assistant:"""
    
    return query_model(prompt)

if __name__ == "__main__":
    print(ask("What is the ruling for Snapcaster Mage regarding flashback?"))