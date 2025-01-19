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

1. **Create Deployments and Services**
   - Use `kubectl apply` to create the Kubernetes resources for your service. Replace `deployment.yml` with the appropriate file for your service:
     ```bash
     kubectl apply -f deployment.yml
     ```

2. **Verify Pod Status**
   - Check the status of your pods to ensure they are running properly:
     ```bash
     kubectl get pods
     ```

---

## Notes

- Repeat these steps for every service contained in the `services` directory.
- Ensure each service has its own `Dockerfile`, `deployment.yml`, and unique configurations as needed.
- For services using custom Docker images or private registries, ensure credentials are configured properly in Kubernetes (e.g., using `imagePullSecrets`).

