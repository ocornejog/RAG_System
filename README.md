# RAG System Deployment Guide

## Project Structure
```
.
├── api/
│   ├── kubernetes/
│   ├── Dockerfile
│   ├── deployment.yaml
│   ├── requirements.txt
│   └── service_calls.py
├── services/
│   ├── chat_service/
│   │   ├── kubernetes/
│   │   ├── __pycache__/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── composite_service/
│   │   ├── kubernetes/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── encode_service/
│   │   ├── kubernetes/
│   │   ├── __pycache__/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── query_service/
│       ├── kubernetes/
│       ├── __pycache__/
│       ├── Dockerfile
│       └── requirements.txt
├── tests/
├── .env
├── exec_commands
└── main.py
```

## Prerequisites
- Docker
- Kubernetes (kubectl)
- Minikube
- Python 3.11+

## Quick Start

1. **Environment Setup**
   ```bash
   # Create and configure .env file
   cp .env.example .env
   # Edit .env file with your configurations
   ```

2. **Deployment**
   ```bash
   # Make exec_commands executable
   chmod +x exec_commands
   
   # Run deployment script
   ./exec_commands
   ```

3. **Start the Application**
   ```bash
   # Start the UI
   python main.py
   ```

## Manual Deployment

If you prefer to deploy manually, follow these steps:

1. **Start Minikube**
   ```bash
   minikube start
   minikube addons enable ingress
   eval $(minikube docker-env)  # For Unix-based systems
   # For Windows: @FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env') DO @%i
   ```

2. **Build Docker Images**
   ```bash
   # Build API Gateway
   docker build -t api-gateway:latest -f api/Dockerfile ./api
   
   # Build Services
   docker build -t chat-service:latest -f services/chat_service/Dockerfile ./services/chat_service
   docker build -t encode-service:latest -f services/encode_service/Dockerfile ./services/encode_service
   docker build -t query-service:latest -f services/query_service/Dockerfile ./services/query_service
   docker build -t composite-service:latest -f services/composite_service/Dockerfile ./services/composite_service
   ```

3. **Deploy to Kubernetes**
   ```bash
   # Create namespace
   kubectl create namespace rag-system
   
   # Deploy services
   kubectl apply -f api/kubernetes/
   kubectl apply -f services/encode_service/kubernetes/
   kubectl apply -f services/query_service/kubernetes/
   kubectl apply -f services/chat_service/kubernetes/
   kubectl apply -f services/composite_service/kubernetes/
   ```

## Verification

Check deployment status:
```bash
# Check pods
kubectl get pods -n rag-system

# Check services
kubectl get services -n rag-system

# Get service URLs
minikube service list -n rag-system
```

## Troubleshooting

1. **Pod Status Issues**
   ```bash
   kubectl describe pod <pod-name> -n rag-system
   kubectl logs <pod-name> -n rag-system
   ```

2. **Service Connectivity**
   ```bash
   kubectl get endpoints -n rag-system
   ```

3. **Common Issues**
   - If pods are in `ImagePullBackOff`, check image names and tags
   - If pods are in `CrashLoopBackOff`, check logs for errors
   - If services aren't accessible, verify service and pod labels match

## Service Ports
- API Gateway: 8000
- Chat Service: 8002
- Query Service: 8001
- Encode Service: 8000
- Composite Service: 8003

## Clean Up
```bash
# Delete all resources
kubectl delete namespace rag-system

# Stop Minikube
minikube stop
```