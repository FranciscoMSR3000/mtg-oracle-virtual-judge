"""
chain.py — RAG chain for MTG Oracle
 
This module implements the Retrieval-Augmented Generation (RAG) pipeline.
It runs on every user query — retrieves relevant rulings from ChromaDB,
constructs a grounded prompt, and generates a response using the fine-tuned model.
 
RAG pattern:
    User question
        → embedding model converts question to vector
        → ChromaDB finds the 5 most similar ruling documents
        → prompt is built with those rulings as context
        → fine-tuned phi-2 generates a grounded response
"""
 
import os
import torch
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
 
# Load API keys and config from .env file
load_dotenv()
 
# ── Model loading ──────────────────────────────────────────────────────────────
# Models are loaded once at startup — loading on every request would be too slow.
# phi-2 is the base model; our LoRA adapter is applied on top.
#
# Why two-step loading?
# We uploaded only the LoRA adapter (~10MB) to Hugging Face, not phi-2 (5.6GB).
# To use the model: load phi-2 base → apply our LoRA adapter on top.
print("Loading model...")
 
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True  # phi-2 uses custom Microsoft code
)
 
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    # Use float16 on GPU (half the memory), float32 on CPU (no CUDA float16 support)
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
 
# Apply our fine-tuned LoRA adapter on top of the frozen phi-2 weights
# The adapter was trained on 76,716 MTG ruling Q&A pairs
model = PeftModel.from_pretrained(base_model, "Razo3000/mtg-oracle")
 
# Set to evaluation mode — disables dropout, freezes all weights
# This is inference only, not training
model.eval()
 
print("✓ Model loaded!")
 
 
def get_collection():
    """
    Connect to the ChromaDB collection containing indexed MTG rulings.
 
    Uses the same embedding model as ingest.py — critical requirement:
    if the embedding model differs between indexing and querying,
    similarity search becomes meaningless (different vector spaces).
 
    Returns:
        chromadb.Collection: the mtg_rulings collection
    """
    client = chromadb.PersistentClient(path="./chroma_db")
 
    # IMPORTANT: must match the model used in ingest.py
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
 
    return client.get_collection(name="mtg_rulings", embedding_function=ef)
 
 
def query_model(prompt: str) -> str:
    """
    Run inference with the fine-tuned model.
 
    Args:
        prompt: formatted prompt string with system context + rulings + question
 
    Returns:
        str: the model's generated response (assistant turn only)
    """
    # Tokenize the prompt — converts text to token IDs the model understands
    inputs = tokenizer(prompt, return_tensors="pt")
 
    # Move inputs to GPU if available — dramatically faster than CPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
 
    # Disable gradient computation — not needed for inference, saves memory
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,   # maximum tokens to generate
            temperature=0.7,      # 0=deterministic, 1=random — 0.7 is balanced
            do_sample=True,       # enable sampling (required for temperature > 0)
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # stop generation at EOS token
        )
 
    # Decode token IDs back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    # Extract only the assistant's response — everything after "Assistant:"
    return response.split("Assistant:")[-1].strip()
 
 
def ask(question: str) -> str:
    """
    Main RAG pipeline — retrieves relevant rulings and generates a grounded answer.
 
    Three-step process:
    1. RETRIEVAL: convert question to vector, find 5 most similar rulings in ChromaDB
    2. AUGMENTATION: build prompt with retrieved rulings as context
    3. GENERATION: fine-tuned model generates response based only on provided rulings
 
    Args:
        question: natural language question about MTG rules or card evaluation
 
    Returns:
        str: grounded answer based on official Scryfall rulings
    """
    collection = get_collection()
 
    # ── Step 1: Retrieval ────────────────────────────────────────────────────
    # ChromaDB converts the question to a vector using the same embedding model,
    # then finds the 5 documents with the highest cosine similarity.
    # This is semantic search — finds relevant rulings even without exact keywords.
    results = collection.query(
        query_texts=[question],
        n_results=5  # retrieve top 5 most relevant rulings
    )
 
    # Join the 5 retrieved documents into a single context block
    context = "\n\n".join(results["documents"][0])
 
    # ── Step 2: Augmentation ─────────────────────────────────────────────────
    # Build the prompt in the same format used during fine-tuning (System/User/Assistant).
    # The "ONLY" instruction prevents hallucination — model must use provided rulings.
    prompt = f"""System: You are MTG Oracle, an expert Magic: The Gathering judge and analyst.
You explain rules interactions clearly and evaluate cards with authority.
Respond based ONLY on the rulings provided below.
If the information is not in the rulings, say so clearly.
 
Relevant rulings:
{context}
 
User: {question}
Assistant:"""
 
    # ── Step 3: Generation ───────────────────────────────────────────────────
    return query_model(prompt)
 
 
# Quick test when running this file directly
if __name__ == "__main__":
    test_questions = [
        "What is the ruling for Snapcaster Mage regarding flashback?",
        "Evaluate Blood Moon for Modern.",
    ]
 
    for q in test_questions:
        print(f"\nQ: {q}")
        print("-" * 60)
        print(ask(q))