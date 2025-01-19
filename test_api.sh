curl http://localhost:8000/

curl -X POST http://localhost:8000/encode_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "La France maintient sa position unique dans le paysage culturel mondial.",
      "L intelligence artificielle continue de transformer notre société.",
      "Le changement climatique représente un défi majeur."
    ]
  }'


curl -X POST http://localhost:8000/search_documents \
  -H "Content-Type: application/json" \
  -d '{
    "query": "intelligence artificielle"
  }'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Que peux-tu me dire sur l IA?"}
    ],
    "temperature": 0.7,
    "max_relevant_chunks": 3
  }'

# 5. Obtenir les statistiques
# Statistiques globales
curl http://localhost:8000/stats

# Statistiques des documents
curl http://localhost:8000/stats/documents

# Statistiques de recherche
curl http://localhost:8000/stats/search

# Statistiques de chat
curl http://localhost:8000/stats/chat

# 6. Logger une recherche (pour les tests)
curl -X POST http://localhost:8000/log/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "intelligence artificielle",
    "success": true,
    "num_results": 3
  }'

# 7. Logger un chat (pour les tests)
curl -X POST http://localhost:8000/log/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Que peux-tu me dire sur l IA?",
    "success": true,
    "duration": 1.5,
    "num_sources": 3
  }'

# Script complet de test
#!/bin/bash

echo "1. Testing root endpoint..."
curl -s http://localhost:8000/

echo -e "\n\n2. Encoding documents..."
curl -s -X POST http://localhost:8000/encode_documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "La France maintient sa position unique dans le paysage culturel mondial.",
      "L intelligence artificielle continue de transformer notre société.",
      "Le changement climatique représente un défi majeur."
    ]
  }'

echo -e "\n\n3. Testing search..."
curl -s -X POST http://localhost:8000/search_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "intelligence artificielle"}'

echo -e "\n\n4. Testing chat..."
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Que peux-tu me dire sur l IA?"}],
    "temperature": 0.7,
    "max_relevant_chunks": 3
  }'

echo -e "\n\n5. Getting statistics..."
echo "Global stats:"
curl -s http://localhost:8000/stats
echo -e "\n\nDocument stats:"
curl -s http://localhost:8000/stats/documents
echo -e "\n\nSearch stats:"
curl -s http://localhost:8000/stats/search
echo -e "\n\nChat stats:"
curl -s http://localhost:8000/stats/chat

echo -e "\n\n6. Logging test search..."
curl -s -X POST http://localhost:8000/log/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "intelligence artificielle",
    "success": true,
    "num_results": 3
  }'

echo -e "\n\n7. Logging test chat..."
curl -s -X POST http://localhost:8000/log/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Que peux-tu me dire sur l IA?",
    "success": true,
    "duration": 1.5,
    "num_sources": 3
  }'

echo -e "\n\nTest complete!"