#!/bin/bash
# Production Deployment Script for Arynoxtech World Model
# Usage: ./deploy/production_deploy.sh [environment]

set -e

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_NAME="arynoxtech-world-model"
REGISTRY=${DOCKER_REGISTRY:-docker.io}
IMAGE_NAME="${REGISTRY}/arynoxtech/world-model"
VERSION=${VERSION:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Arynoxtech World Model Deployment${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}========================================${NC}"

# Pre-deployment checks
echo -e "${YELLOW}Running pre-deployment checks...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if kubectl is installed (for Kubernetes deployment)
if [[ "${ENVIRONMENT}" == "kubernetes" ]] && ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl is not installed. Please install kubectl first.${NC}"
    exit 1
fi

# Build and test
echo -e "${YELLOW}Building and testing...${NC}"
python -m pytest tests/ -v --tb=short

if [ $? -ne 0 ]; then
    echo -e "${RED}Tests failed. Aborting deployment.${NC}"
    exit 1
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${VERSION} .

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed. Aborting deployment.${NC}"
    exit 1
fi

# Deploy based on environment
case ${ENVIRONMENT} in
    "production")
        echo -e "${YELLOW}Deploying to production...${NC}"
        
        # Push to registry
        docker push ${IMAGE_NAME}:${VERSION}
        
        # Deploy using docker-compose
        docker-compose -f docker-compose.prod.yml up -d
        
        echo -e "${GREEN}✓ Production deployment completed!${NC}"
        ;;
    
    "staging")
        echo -e "${YELLOW}Deploying to staging...${NC}"
        
        # Push to registry
        docker push ${IMAGE_NAME}:${VERSION}
        
        # Deploy using docker-compose
        docker-compose -f docker-compose.staging.yml up -d
        
        echo -e "${GREEN}✓ Staging deployment completed!${NC}"
        ;;
    
    "kubernetes")
        echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
        
        # Apply Kubernetes configurations
        kubectl apply -f k8s/configmap.yaml
        kubectl apply -f k8s/secret.yaml
        kubectl apply -f k8s/deployment.yaml
        kubectl apply -f k8s/service.yaml
        kubectl apply -f k8s/ingress.yaml
        
        echo -e "${GREEN}✓ Kubernetes deployment completed!${NC}"
        ;;
    
    "local")
        echo -e "${YELLOW}Deploying locally...${NC}"
        
        # Run locally with Docker
        docker run -d \
            -p 8501:8501 \
            -e GROQ_API_KEY=${GROQ_API_KEY} \
            -v $(pwd)/user_data:/app/user_data \
            ${IMAGE_NAME}:${VERSION}
        
        echo -e "${GREEN}✓ Local deployment completed!${NC}"
        echo -e "${GREEN}Application running at: http://localhost:8501${NC}"
        ;;
    
    *)
        echo -e "${RED}Unknown environment: ${ENVIRONMENT}${NC}"
        echo -e "${YELLOW}Usage: $0 [production|staging|kubernetes|local]${NC}"
        exit 1
        ;;
esac

# Post-deployment checks
echo -e "${YELLOW}Running post-deployment health checks...${NC}"
sleep 10  # Wait for application to start

# Check if application is running
if curl -f -s http://localhost:8501 > /dev/null; then
    echo -e "${GREEN}✓ Application is healthy!${NC}"
else
    echo -e "${YELLOW}⚠ Application may still be starting up. Please check manually.${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"