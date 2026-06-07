"""
ingest.py — Data ingestion pipeline for MTG Oracle
 
This script runs ONCE to build the ChromaDB vector database from Scryfall data.
It downloads all official MTG rulings and indexes them as embeddings so the
RAG system can retrieve relevant rulings at query time.
 
Run with: python src/ingest.py
"""
 
import os
import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
 
# Load environment variables from .env file
load_dotenv()
 
# ── Scryfall API configuration ─────────────────────────────────────────────────
# Scryfall requires a custom User-Agent header to identify your application.
# Without it, all requests return 400 Bad Request (per their API policy).
HEADERS = {
    "User-Agent": "MTG-Oracle/1.0 (educational project)",
    "Accept": "application/json"
}
 
 
def download_rulings():
    """
    Download all MTG rulings and card data from Scryfall bulk data endpoint.
 
    Scryfall provides daily database snapshots — much more efficient than
    making individual API calls for each card (37,000+ cards × 2 requests = 74,000 calls).
 
    Returns:
        tuple: (all_rulings, all_cards) — raw JSON lists from Scryfall
    """
    print("Fetching bulk data index from Scryfall...")
    bulk_response = requests.get("https://api.scryfall.com/bulk-data", headers=HEADERS)
    bulk_data = bulk_response.json()
 
    # Extract download URLs for the two datasets we need
    rulings_url = next(i['download_uri'] for i in bulk_data['data'] if i['type'] == 'rulings')
    oracle_url  = next(i['download_uri'] for i in bulk_data['data'] if i['type'] == 'oracle_cards')
 
    print("Downloading rulings (24MB)...")
    rulings = requests.get(rulings_url, headers=HEADERS).json()
 
    print("Downloading oracle cards (165MB)...")
    cards = requests.get(oracle_url, headers=HEADERS).json()
 
    return rulings, cards
 
 
def build_documents(rulings, cards):
    """
    Join rulings with card data using oracle_id as the key.
 
    Each card has an oracle_id that uniquely identifies its rules text.
    All printings of the same card share the same oracle_id.
    This is equivalent to a SQL JOIN:
        SELECT cards.name, rulings.comment
        FROM cards JOIN rulings ON cards.oracle_id = rulings.oracle_id
 
    Args:
        rulings: list of ruling objects from Scryfall
        cards:   list of card objects from Scryfall
 
    Returns:
        tuple: (documents, ids, metadatas) ready for ChromaDB upsert
    """
 
    # Index cards by oracle_id for O(1) lookup during the join
    cards_by_id = {}
    for card in cards:
        oracle_id = card.get('oracle_id')
        if oracle_id:
            cards_by_id[oracle_id] = {
                "name":        card.get('name', ''),
                "type_line":   card.get('type_line', ''),
                "oracle_text": card.get('oracle_text', ''),
                "mana_cost":   card.get('mana_cost', ''),
            }
 
    # Group rulings by oracle_id (one card can have multiple rulings)
    rulings_by_card = {}
    for ruling in rulings:
        oracle_id = ruling.get('oracle_id')
        if oracle_id:
            if oracle_id not in rulings_by_card:
                rulings_by_card[oracle_id] = []
            rulings_by_card[oracle_id].append(ruling['comment'])
 
    print(f"✓ Cards indexed: {len(cards_by_id)}")
    print(f"✓ Cards with rulings: {len(rulings_by_card)}")
 
    # Build ChromaDB documents — each ruling becomes one document
    documents = []
    ids       = []
    metadatas = []
 
    for oracle_id, card_rulings in rulings_by_card.items():
        card = cards_by_id.get(oracle_id)
        if not card:
            continue
 
        for i, ruling in enumerate(card_rulings):
            # Skip rulings that are too short to be informative
            # (e.g., "See rule 702.19." adds no value for RAG retrieval)
            if len(ruling) < 30:
                continue
 
            # Convert each ruling into a descriptive text document.
            # ChromaDB stores text as embeddings — the embedding model needs
            # natural language, not raw JSON or table rows.
            doc = f"""
            Card: {card['name']}
            Type: {card['type_line']}
            Oracle Text: {card['oracle_text']}
            Ruling: {ruling}
            """
 
            documents.append(doc)
            ids.append(f"{oracle_id}_{i}")          # unique ID per ruling
            metadatas.append({
                "card_name": card['name'],
                "type_line": card['type_line'],
            })
 
    return documents, ids, metadatas
 
 
def ingest_data():
    """
    Main ingestion pipeline — downloads data and indexes it into ChromaDB.
 
    ChromaDB stores each document as a vector embedding, enabling semantic
    search: finding documents by meaning rather than exact keyword matching.
 
    The embedding model (all-MiniLM-L6-v2) converts each text document into
    a 384-dimensional vector. Similar documents produce similar vectors.
    """
    # Download data from Scryfall
    rulings, cards = download_rulings()
 
    # Build document corpus
    documents, ids, metadatas = build_documents(rulings, cards)
    print(f"\n✓ Built {len(documents)} documents")
 
    # Connect to ChromaDB with persistent storage
    # PersistentClient saves to disk — data survives program restarts
    client = chromadb.PersistentClient(path="./chroma_db")
 
    # Define the embedding model — MUST be the same model used in chain.py
    # Using different models for indexing vs querying breaks similarity search
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # lightweight model, 384-dim embeddings
    )
 
    # Get or create the collection (like a table in a regular database)
    collection = client.get_or_create_collection(
        name="mtg_rulings",
        embedding_function=ef
    )
 
    # Insert in batches of 500 to avoid memory issues with large datasets
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_docs  = documents[i:i+batch_size]
        batch_ids   = ids[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
 
        # upsert = insert if new, update if exists
        # Allows re-running this script safely without duplicates
        collection.upsert(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas
        )
        print(f"  Indexed {min(i+batch_size, len(documents))}/{len(documents)}")
 
    print("\n✓ All rulings indexed in ChromaDB")
    print(f"  Database saved to ./chroma_db/")
 
 
if __name__ == "__main__":
    ingest_data()