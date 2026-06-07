import os
import json
import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

def download_rulings():
    """Descarga los rulings de Scryfall bulk data"""
    headers = {
        "User-Agent": "MTG-Oracle/1.0 (educational project)",
        "Accept": "application/json"
    }
    
    print("Fetching bulk data index from Scryfall...")
    bulk_response = requests.get("https://api.scryfall.com/bulk-data", headers=headers)
    bulk_data = bulk_response.json()
    
    # Obtener URLs
    rulings_url = next(i['download_uri'] for i in bulk_data['data'] if i['type'] == 'rulings')
    oracle_url  = next(i['download_uri'] for i in bulk_data['data'] if i['type'] == 'oracle_cards')
    
    print("Downloading rulings (24MB)...")
    rulings = requests.get(rulings_url, headers=headers).json()
    
    print("Downloading oracle cards (165MB)...")
    cards = requests.get(oracle_url, headers=headers).json()
    
    return rulings, cards

def build_documents(rulings, cards):
    """Cruza rulings con cartas y construye documentos de texto"""
    
    # Indexar cartas por oracle_id
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
    
    # Agrupar rulings por oracle_id
    rulings_by_card = {}
    for ruling in rulings:
        oracle_id = ruling.get('oracle_id')
        if oracle_id:
            if oracle_id not in rulings_by_card:
                rulings_by_card[oracle_id] = []
            rulings_by_card[oracle_id].append(ruling['comment'])
    
    # Construir documentos
    documents = []
    ids       = []
    metadatas = []
    
    for oracle_id, card_rulings in rulings_by_card.items():
        card = cards_by_id.get(oracle_id)
        if not card:
            continue
        
        for i, ruling in enumerate(card_rulings):
            if len(ruling) < 30:
                continue
            
            doc = f"""
            Card: {card['name']}
            Type: {card['type_line']}
            Oracle Text: {card['oracle_text']}
            Ruling: {ruling}
            """
            
            documents.append(doc)
            ids.append(f"{oracle_id}_{i}")
            metadatas.append({
                "card_name": card['name'],
                "type_line": card['type_line'],
            })
    
    return documents, ids, metadatas

def ingest_data():
    rulings, cards = download_rulings()
    documents, ids, metadatas = build_documents(rulings, cards)
    
    print(f"✓ Built {len(documents)} documents")
    
    # Conectar con ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="mtg_rulings",
        embedding_function=ef
    )
    
    # Insertar en batches de 500
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_docs  = documents[i:i+batch_size]
        batch_ids   = ids[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        print(f"  Indexed {min(i+batch_size, len(documents))}/{len(documents)}")
    
    print("✓ All rulings indexed in ChromaDB")

if __name__ == "__main__":
    ingest_data()