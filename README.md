# 🃏 MTG Oracle

A Retrieval-Augmented Generation (RAG) system for Magic: The Gathering rules and card evaluation. Combines a **fine-tuned phi-2 model** with a **ChromaDB vector database** of 76,704 official Scryfall rulings to provide accurate, grounded answers about MTG rules interactions and card assessments.

> ⚠️ **Disclaimer:** This project uses official Scryfall rulings as its knowledge base. Always verify with a certified judge for tournament play. The fine-tuning dataset is built from public Scryfall data under their [terms of use](https://scryfall.com/docs/terms).

---

## 🎯 What it does

Instead of relying on a generic LLM that may hallucinate MTG rules, MTG Oracle retrieves the actual official ruling before generating a response:

```
User: "What is the ruling for Snapcaster Mage regarding flashback?"

MTG Oracle: "Regarding Snapcaster Mage: You must still follow any timing
             restrictions and permissions, including those based on the
             card's type. For instance, you can only cast a sorcery this
             way during your main phase when the stack is empty."
```

---

## 🏗️ Architecture

```
Fine-tuning Pipeline (Google Colab)
────────────────────────────────────
Scryfall API → 76,716 Q&A pairs → LoRA fine-tuning → phi-2 specialized model
                                                              ↓
                                               Hugging Face Hub (Razo3000/mtg-oracle)

RAG Pipeline (local)
─────────────────────
User question
      ↓
Embedding model (all-MiniLM-L6-v2) converts question to vector
      ↓
ChromaDB finds top-5 most semantically similar rulings
      ↓
Prompt: system instruction + retrieved rulings + user question
      ↓
Fine-tuned phi-2 generates grounded response
      ↓
Answer based on official Scryfall rulings
```

### Why RAG + Fine-tuning together?

| Approach | What it provides |
|---|---|
| Fine-tuning alone | Correct style and MTG vocabulary, but may hallucinate ruling details |
| RAG alone | Accurate data retrieval, but generic response style |
| **Fine-tuning + RAG** | Expert MTG judge style + factually grounded answers |

Fine-tuning taught the model **how to respond** (tone, format, authority).
RAG provides **what to respond with** (the actual official rulings).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Base LLM | microsoft/phi-2 (2.8B parameters) |
| Fine-tuning | LoRA via Hugging Face PEFT |
| Model hosting | Hugging Face Hub |
| Vector database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Interface | Streamlit |
| Data source | Scryfall Bulk Data API |
| Training environment | Google Colab (T4 GPU) |

---

## 📁 Project Structure

    mtg-oracle/
    ├── src/
    │   ├── ingest.py              # Downloads Scryfall data and indexes into ChromaDB
    │   └── chain.py               # RAG pipeline — retrieval + prompt + generation
    ├── app.py                     # Streamlit chat interface
    ├── MTG_Oracle_Training.ipynb  # Google Colab fine-tuning notebook
    ├── .env                       # API keys (not included in repo)
    └── README.md

---

## 📊 Dataset

**76,704 training pairs** built from Scryfall bulk data:

- **76,692** automated ruling Q&A pairs (from 19,623 cards with official rulings)
- **12** hand-crafted card evaluation pairs (teaching evaluation style and scoring)

Each pair follows ChatML format:
```json
{
  "messages": [
    {"role": "system",    "content": "You are MTG Oracle, an expert judge..."},
    {"role": "user",      "content": "What is the ruling for Snapcaster Mage regarding flashback?"},
    {"role": "assistant", "content": "Regarding Snapcaster Mage: If you cast a spell with flashback..."}
  ]
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 recommended (3.14 works but runs on CPU only — no CUDA support yet)
- ~10GB disk space for phi-2 model weights + ChromaDB

### 1. Clone the repository

```bash
git clone https://github.com/FranciscoMSR3000/mtg-oracle.git
cd mtg-oracle
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers peft
pip install chromadb sentence-transformers
pip install streamlit python-dotenv requests
```

### 3. Configure environment variables

Create a `.env` file in the root folder:

    HUGGINGFACEHUB_API_TOKEN=hf_...

### 4. Build the ChromaDB vector database

Downloads 76,704 official rulings from Scryfall and indexes them as embeddings.
Run once — takes ~5 minutes depending on internet speed.

```bash
python src/ingest.py
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

> **Note:** The first run downloads phi-2 (~5.6GB) from Hugging Face and loads it into memory.
> On CPU this takes 3-5 minutes. With a CUDA GPU (Python 3.11 + PyTorch CUDA) it loads in ~30 seconds.

---

## ⚡ Performance

| Hardware | Load time | Response time |
|---|---|---|
| CPU only (Python 3.14) | ~3-5 min | ~5-10 min per response |
| NVIDIA GPU + CUDA (Python 3.11) | ~30 sec | ~5-10 sec per response |

For production use, the model should be deployed on a GPU server and accessed via API rather than loading it locally.

---

## 🔬 Fine-tuning Details

The model was fine-tuned on Google Colab using a T4 GPU (~3 hours):

| Parameter | Value |
|---|---|
| Base model | microsoft/phi-2 |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA rank (r) | 8 |
| Trainable parameters | 2,621,440 / 2,782,305,280 (0.09%) |
| Training examples | 4,750 (sampled from 76,716) |
| Epochs | 3 |
| Final training loss | 0.80 |
| Final validation loss | 0.78 |

The LoRA adapter (~10MB) is published at [huggingface.co/Razo3000/mtg-oracle](https://huggingface.co/Razo3000/mtg-oracle).
Loading requires the phi-2 base model + this adapter applied on top.

---

## 🗺️ Roadmap

- [ ] Python 3.11 + CUDA support for GPU inference (10-50x faster)
- [ ] Metadata filtering in ChromaDB (filter by card name, set, format)
- [ ] Deploy model as API endpoint (FastAPI + GPU server)
- [ ] Deploy interface to Streamlit Cloud
- [ ] Expand evaluation dataset (currently only 12 hand-crafted examples)
- [ ] RAGAS evaluation suite to measure retrieval and generation quality

---

## 👤 Author

Francisco — [GitHub](https://github.com/FranciscoMSR3000) · [Hugging Face](https://huggingface.co/Razo3000)

*Built as part of a GenAI portfolio demonstrating fine-tuning + RAG on a specialized domain.*
