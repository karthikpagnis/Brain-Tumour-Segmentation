# Deployment Guide

## Quick Start

### Local Development

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Create mock dataset (for testing)
python scripts/download_data.py --create_mock --num_cases 5

# 3. Train model
python training/train.py --epochs 10 --batch_size 8

# 4. Run API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Run web UI
cd ui && npm install && npm start
```

### API Access
- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health
- Model Info: http://localhost:8000/api/model-info

### Web UI
- Open: http://localhost:3000
- Upload MRI scan
- View segmentation results
- Download NIfTI prediction

## Docker Deployment

### Build Docker Image

```bash
docker build -t brain-tumor-seg:latest .
```

### Run with Docker Compose

```bash
docker-compose up --build
```

Services:
- API: http://localhost:8000
- Web UI: http://localhost:3000
- Database: PostgreSQL (localhost:5432)

## Production Deployment

### Environment Variables

```bash
# .env
DEVICE=cuda
API_WORKERS=4
LOG_LEVEL=INFO
MODEL_PATH=checkpoints/attention_unet_v1_best.pth
```

### Scaling

**Option 1: Gunicorn + Uvicorn**
```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Option 2: Docker Scaling**
```bash
docker-compose up -d --scale api=3
```

**Option 3: Kubernetes**
- Create deployment.yaml
- Define resource requests/limits
- Use horizontal pod autoscaling

### GPU Allocation

```yaml
# docker-compose.yml excerpt
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [compute, utility]
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=. --cov-report=html

# Integration test
curl -X POST \
  -F "file=@sample.nii.gz" \
  http://localhost:8000/api/predict
```

## Monitoring

### Logs

```bash
# API logs
tail -f outputs/training.log

# Docker logs
docker-compose logs -f api
```

### TensorBoard

```bash
tensorboard --logdir=outputs/logs/tensorboard
```

### Metrics

- CPU/Memory: Monitor with `docker stats`
- Inference time: Logged in API responses
- Model accuracy: Tracked in TensorBoard

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU out of memory | Reduce batch size in config.py |
| Slow inference | Use smaller patch size or GPU with more memory |
| API won't start | Check if port 8000 is free |
| UI can't connect | Verify CORS settings in config.py |
| Model not found | Download BraTS dataset or create mock data |

## Security

- Use HTTPS in production
- Validate file uploads (size, format)
- Implement rate limiting
- Use API keys for production
- Store model checkpoints securely

## Performance Optimization

1. **Inference Batching**: Process multiple images together
2. **Model Quantization**: Reduce model size (optional)
3. **Caching**: Cache preprocessing results
4. **Load Balancing**: Distribute across GPUs
5. **Async Processing**: Use background tasks for heavy operations

## Backup & Recovery

```bash
# Backup model and data
tar -czf backup_$(date +%Y%m%d).tar.gz \
  checkpoints/ data/ outputs/

# Restore from backup
tar -xzf backup_20250411.tar.gz
```
