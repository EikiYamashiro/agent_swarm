# Agent Swarm: Infinitepay Knowledge Base

A simple RAG (Retrieval Augmented Generation) system that answers questions about Infinitepay products and services by retrieving relevant information from their website.

## How it works

1. **Knowledge Agent**
   - Downloads and indexes specified Infinitepay pages
   - Uses TF-IDF to find relevant text passages for each query
   - Returns both answer text and source URLs

2. **API**
   - POST /chat endpoint accepts questions
   - Returns retrieved passages and sources

## Running locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn backend.main:app --reload
```

3. Ask questions:
```bash
# Example: ask about Maquininha Smart fees
curl -X POST "http://localhost:8000/swarm" \
     -H "Content-Type: application/json" \
     -d '{"message":"What are the fees of the Maquininha Smart"}'
```

## Example queries

- "What are the fees of the Maquininha Smart?"
- "How can I use my phone as a card machine?"
- "What are the rates for debit and credit card transactions?"
- "Tell me about Pix parcelado"

## Technical notes

- Uses scikit-learn's TfidfVectorizer for passage retrieval
- Chunks pages into ~800 char passages
- Returns top 3 most relevant passages per query
- Simple but effective for factual queries about products
