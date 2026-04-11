# Cloud Deployment Guide

## Overview

Deploy your Brain Tumor Segmentation model to production cloud infrastructure.

---

## AWS Deployment (Recommended for US/Global)

### Step 1: Set Up AWS Account

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Format (json)
```

### Step 2: Create EC2 Instance

```bash
# Launch GPU instance (Amazon Linux 2 AMI)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type g4dn.2xlarge \
    --key-name your-key-pair \
    --security-groups allow-ssh,allow-http,allow-https \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=brain-tumor-seg}]'
```

**Instance Types:**
- `g4dn.2xlarge`: 1× NVIDIA T4 GPU, 8 CPU, 32 GB RAM (~$1.21/hr)
- `g4dn.12xlarge`: 4× NVIDIA T4 GPU (~$4.03/hr)
- `p3.2xlarge`: 1× NVIDIA V100 GPU (~$3.06/hr)

### Step 3: SSH into Instance

```bash
# Get public IP
aws ec2 describe-instances --query 'Reservations[0].Instances[0].PublicIpAddress'

# SSH into instance
ssh -i your-key.pem ec2-user@<public-ip>
```

### Step 4: Set Up Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10 python3.10-venv git -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/Brain-Tumour-Segmentation.git
cd Brain-Tumour-Segmentation

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Download BraTS Dataset

```bash
# Option A: Transfer from local machine
scp -i your-key.pem -r data/BraTS ec2-user@<public-ip>:~/Brain-Tumour-Segmentation/data/

# Option B: Download directly on server
# (Register at https://www.med.upenn.edu/cbica/brats2021/)
# Download link, then:
wget <download-url> -O brats.zip
unzip brats.zip -d data/BraTS
```

### Step 6: Start Training

```bash
# Run training in background
nohup python training/train.py \
    --experiment_name aws_attention_unet \
    --epochs 100 \
    > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Step 7: Deploy as API

```bash
# Start FastAPI server
nohup python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    > api.log 2>&1 &

# Test API
curl http://<public-ip>:8000/api/health
```

### Step 8: Set Up Elastic IP (Static URL)

```bash
# Allocate Elastic IP
aws ec2 allocate-address --domain vpc

# Associate with instance
aws ec2 associate-address \
    --instance-id <instance-id> \
    --allocation-id <allocation-id>
```

### Step 9: Upload Trained Model

```bash
# Transfer trained model to S3
aws s3 cp checkpoints/attention_unet_v1_best.pth \
    s3://your-bucket/models/

# Create CloudFront distribution for fast downloads
```

---

## Google Cloud Platform (GCP) Deployment

### Step 1: Set Up GCP Project

```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash

# Initialize
gcloud init
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Create Compute Instance

```bash
# Create VM with GPU
gcloud compute instances create brain-tumor-seg \
    --image-family=torch-gpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --zone=us-central1-a \
    --boot-disk-size=100
```

### Step 3: Connect and Set Up

```bash
# SSH into instance
gcloud compute ssh brain-tumor-seg --zone=us-central1-a

# Inside VM, repeat AWS setup steps:
git clone <repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Use Cloud Storage for Data

```python
# In config.py
from google.cloud import storage
import os

# Mount GCS bucket
os.system('gsutil -m cp -r gs://your-bucket/BraTS data/')

# Or use GCSFuse for transparent access
sudo apt-get install gcsfuse
mkdir -p ~/mnt/brats
gcsfuse your-bucket ~/mnt/brats
```

### Step 5: Deploy with Cloud Run

```bash
# Create Dockerfile for Cloud Run
cat > Dockerfile.clouddeploy <<EOF
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
EOF

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/brain-tumor-seg

# Deploy to Cloud Run
gcloud run deploy brain-tumor-seg \
    --image gcr.io/YOUR_PROJECT/brain-tumor-seg \
    --memory=16Gi \
    --timeout=3600 \
    --allow-unauthenticated
```

---

## Microsoft Azure Deployment

### Step 1: Set Up Azure Account

```bash
pip install azure-cli

az login
az account show
```

### Step 2: Create Resource Group

```bash
az group create \
    --name brain-tumor-rg \
    --location eastus
```

### Step 3: Create Virtual Machine

```bash
az vm create \
    --resource-group brain-tumor-rg \
    --name brain-tumor-vm \
    --image UbuntuLTS \
    --size Standard_NC6s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys
```

### Step 4: Install NVIDIA GPU Support

```bash
# SSH into VM
az vm open-port --port 8000 --resource-group brain-tumor-rg --name brain-tumor-vm

# Inside VM:
sudo apt update
sudo apt install -y nvidia-driver-525
sudo reboot

# Verify GPU
nvidia-smi
```

### Step 5: Deploy with Container Instances

```bash
# Build Docker image
docker build -t brain-tumor-seg:latest .

# Push to Azure Container Registry
az acr create --resource-group brain-tumor-rg --name braintumorseg --sku Basic

docker tag brain-tumor-seg:latest braintumorseg.azurecr.io/brain-tumor-seg:latest
docker push braintumorseg.azurecr.io/brain-tumor-seg:latest

# Deploy
az container create \
    --resource-group brain-tumor-rg \
    --name brain-tumor-api \
    --image braintumorseg.azurecr.io/brain-tumor-seg:latest \
    --cpu 4 --memory 16 \
    --ports 8000
```

---

## Docker Compose Production Setup

### Step 1: Create Production Environment File

```bash
cat > .env.production <<EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=checkpoints/attention_unet_v1_best.pth
DEVICE=cuda

# Database
POSTGRES_DB=brain_tumor
POSTGRES_USER=admin
POSTGRES_PASSWORD=super_secure_password_change_me

# Logging
LOG_LEVEL=INFO

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
EOF
```

### Step 2: Production Docker Compose

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - DEVICE=cuda
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [compute, utility]
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api
    restart: always

  database:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always

volumes:
  postgres_data:
```

### Step 3: HTTPS with Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot certonly --standalone -d yourdomain.com

# Update nginx.conf with certificate paths
sudo docker-compose up -d
```

---

## Kubernetes Deployment (Advanced)

### Step 1: Create Kubernetes Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brain-tumor-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brain-tumor-api
  template:
    metadata:
      labels:
        app: brain-tumor-api
    spec:
      containers:
      - name: api
        image: gcr.io/your-project/brain-tumor-seg:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: brain-tumor-api-service
spec:
  selector:
    app: brain-tumor-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Step 2: Deploy to Kubernetes

```bash
# Apply manifest
kubectl apply -f kubernetes-manifest.yaml

# Check deployment status
kubectl get deployment
kubectl get service
kubectl logs -f deployment/brain-tumor-api
```

---

## Monitoring & Logging

### CloudWatch (AWS)

```bash
# View logs
aws logs tail /aws/ec2/brain-tumor-seg

# Create alarms
aws cloudwatch put-metric-alarm \
    --alarm-name gpu-high-memory \
    --alarm-description "Alert when GPU memory > 90%" \
    --metric-name GPUMemoryUsage \
    --threshold 90
```

### Prometheus & Grafana

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'brain-tumor-api'
    static_configs:
      - targets: ['localhost:8000']
```

```bash
# Start Prometheus
docker run -d -p 9090:9090 -v prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# Start Grafana
docker run -d -p 3000:3000 grafana/grafana
```

---

## Cost Optimization

### Reserved Instances (AWS)

```bash
# Purchase 1-year reserved instance for 30% savings
aws ec2 describe-reserved-instances-offerings \
    --instance-type g4dn.2xlarge \
    --offering-class standard
```

### Spot Instances (For Training)

```bash
# Use spot instances (75% cheaper) for training
aws ec2 run-instances \
    --instance-type g4dn.2xlarge \
    --instance-market-options "MarketType=spot"
```

### Auto-Scaling

```bash
# Scale down during off-peak hours
aws autoscaling put-scheduled-action \
    --auto-scaling-group-name brain-tumor-asg \
    --scheduled-action-name scale-down \
    --recurrence "0 22 * * *" \
    --min-size 1
```

---

## Estimated Costs (Monthly)

| Platform | Configuration | Cost/Month |
|----------|---|---|
| **AWS** | 1× g4dn.2xlarge (predict) | $293 |
| **AWS** | 1× g4dn.2xlarge (24/7 training) | $880 |
| **GCP** | 1× n1-standard-8 + T4 GPU | $275 |
| **Azure** | 1× NC6s v3 (Standard_NC6s_v3) | $260 |

---

## Checklist

- [ ] Create cloud account
- [ ] Launch GPU instance
- [ ] Clone repository
- [ ] Download BraTS dataset
- [ ] Configure training parameters
- [ ] Start training
- [ ] Monitor progress
- [ ] Deploy as API
- [ ] Set up HTTPS
- [ ] Configure monitoring
- [ ] Test inference
- [ ] Set up auto-scaling

---

## Next Steps

1. ✅ Train on full BraTS
2. ✅ Deploy to cloud
3. → **Add Advanced Features**
4. → **Publish Results**

