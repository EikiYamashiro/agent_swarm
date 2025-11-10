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

## Running with Docker (Recommended)

1. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```
2. Build and run containers:

#### Windows (PowerShell)
```bash
docker-compose up --build
```

#### Mac / Linux (Terminal)
```bash
docker compose up --build
```

3. Access the application:
   - Frontend (Streamlit): http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

To stop:
```bash
docker-compose down
```

## Running locally (Development)

1. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`

4. Run the backend:
```bash
uvicorn backend.main:app --reload
```

5. Run the frontend (in another terminal):
```bash
streamlit run frontend/app.py
```

## Testing the API

Example API call:
```bash
# Ask about Maquininha Smart fees
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
- Uses Gemini for response generation
- Containerized with Docker for easy deployment

## Troubleshooting

Common issues and solutions:

1. **Container not starting**
   - Check Docker logs: `docker-compose logs -f`
   - Verify environment variables in `.env`
   - Ensure ports 8000 and 8501 are available

2. **API Authentication Error**
   - Confirm GEMINI_API_KEY is set correctly in `.env`
   - Verify the key has proper permissions

3. **Frontend Can't Connect**
   - Check if backend container is running
   - Verify BACKEND_URL in docker-compose.yml

For development debugging, you can enter the container:
```bash
docker-compose exec app bash
```
