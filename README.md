# Deployment Guide for Services in Kubernetes

This guide outlines the steps required to build, run, and deploy each service contained in the `services` directory. Follow these steps for every service to ensure proper configuration and deployment.

---

## Steps to Build and Run the Container

1. **Prepare the Dockerfile**
   - Save the `Dockerfile` in the root directory of your project.

2. **Navigate to the Project Directory**
   - Open a terminal and navigate to the directory containing the `Dockerfile`.

3. **Build the Docker Image**
   - Execute the following command to build the Docker image:
     ```bash
     docker build -t encode-service .
     ```

4. **Run the Docker Container**
   - Execute the following command to run a container based on the built image:
     ```bash
     docker run -d -p 8000:8000 encode-service
     ```

---

## Steps to Deploy on Kubernetes

1. **Configure Docker for Minikube (Windows)**
   - Execute the following command to configure your terminal to use the Docker environment in Minikube:
     ```cmd
     minikube docker-env --shell=cmd
     ```
   - Then, run this command to apply the environment variables:
     ```cmd
     @FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i
     ```

2. **Verify Docker Images in Minikube**
   - After configuring Docker for Minikube, ensure your images are available:
     ```bash
     docker images
     ```

3. **Create Deployments and Services**
   - Use `kubectl apply` to create the Kubernetes resources for your service. Replace `deployment.yaml` with the appropriate file for your service:
     ```bash
     kubectl apply -f deployment.yaml
     ```

4. **Verify Pod Status**
   - Check the status of your pods to ensure they are running properly:
     ```bash
     kubectl get pods
     ```

5. **Verify Services in Kubernetes**
   - Check the services running in your Kubernetes cluster:
     ```bash
     kubectl get services
     ```

6. **Access the Service via Minikube**
   - Use the following command to access the service in your browser or get a URL to test it:
     ```bash
     minikube service chat-service
     ```

---

## Notes

- Repeat these steps for every service contained in the `services` directory.
- Ensure each service has its own `Dockerfile`, `deployment.yaml`, and unique configurations as needed.
- For services using custom Docker images or private registries, ensure credentials are configured properly in Kubernetes (e.g., using `imagePullSecrets`).


