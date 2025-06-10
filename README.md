# MLOps Assessment - Image Classification Model

This repository contains a machine learning model for image classification, deployed as a Docker container with a FastAPI service. The model is based on a PyTorch model converted to ONNX format for efficient inference using GPU acceleration.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker with NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA GPU with CUDA support
- Python 3.8 or higher
- Git
- Cerebrium account and API key (for deployment)

### GPU Requirements

- CUDA 11.8 or higher
- cuDNN 8.6 or higher
- NVIDIA GPU with compute capability 7.0 or higher

## Project Structure

```
.
├── app.py              # FastAPI application
├── model.py            # ONNX model wrapper with preprocessing
├── pytorch_model.py    # PyTorch model definition
├── convert_to_onnx.py  # Script to convert PyTorch model to ONNX
├── test.py            # Unit tests for local model
├── test_server.py     # Tests for Cerebrium deployment
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── README.md           # Project documentation
├── pytorch_model_weights.pth         # PyTorch model weights
├── schema.py         # Prediction Response structure definition
├── test_images         # Test images for local testing
│   ├── n01440764_tench.jpeg
│   └── n01667114_mud_turtle.JPEG

```

## Deliverables

### 1. Model Conversion (`convert_to_onnx.py`)

- Converts PyTorch model to ONNX format
- Handles model optimization
- Saves model in ONNX format
- Usage:

```bash
python convert_to_onnx.py
```

### 2. Model Implementation (`model.py`)

Contains 1 main class and 1 method:

#### ONNXClassifier

- Handles ONNX model loading and inference
- Manages model session and input/output names
- Provides prediction interface
- GPU-accelerated inference using ONNX Runtime
- Key methods:
  - `__init__`: Loads ONNX model with GPU provider
  - `predict`: Runs inference on input images

#### ImagePreprocessor

- Handles image preprocessing
- Converts images to model input format
- Implements normalization
- Key methods:
  - `preprocess_image`: Converts image bytes to tensor
  - `normalize`: Applies model-specific normalization

### 3. Local Testing (`test.py`)

Comprehensive test suite for local model:

- Model Initialization Tests

  - Verifies model loading
  - Checks GPU availability
  - Validates input/output configurations
- Image Preprocessing Tests

  - Validates image conversion
  - Tests normalization
  - Checks tensor shapes
- Prediction Tests

  - Tests output shapes
  - Verifies prediction values
  - Checks performance metrics
  - GPU utilization monitoring
- API Endpoint Tests

  - Tests health endpoint
  - Validates prediction endpoint
  - Checks error handling

Run tests:

```bash
python -m unittest test.py -v
```

### 4. Cerebrium Deployment

Required files and configurations:

1. `Dockerfile`

   - Base image: NVIDIA CUDA 11.8
   - Dependencies installation
   - Model file copying
   - Port configuration
   - GPU runtime configuration
2. `requirements.txt`

   - FastAPI
   - ONNX Runtime GPU
   - Pillow
   - Other dependencies
3. Deployment Steps:

```bash
# Build Docker image
docker build -t my-onnx-app .

# Push to Cerebrium
cerebrium deploy my-onnx-app
```

### 5. Deployment Testing (`test_server.py`)

Tests for Cerebrium deployment:

- Basic Functionality Tests

  - Image prediction
  - Class ID verification
  - Response time checks
  - GPU utilization monitoring
- Platform Monitoring Tests

  - API availability
  - Response times
  - Error handling
  - Load testing
  - GPU performance metrics
- Custom Test Suite

  - Batch processing
  - Edge cases
  - Performance metrics
  - GPU memory usage

Usage:

```bash
# Test single image
python test_server.py --image path/to/image.jpg

# Run custom tests
python test_server.py --run-tests
```

## Quick Start

1. Clone the repository:

```bash
git clone <repository-url>
cd mtailor_mlops_assessment
```

2. Build the Docker image:

```bash
docker build -t my-onnx-app .
```

3. Run the container with GPU support:

```bash
docker run --gpus all -p 8000:8000 my-onnx-app
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
    "status": "healthy",
    "model_loaded": true,
    "gpu_available": true
}
```

### Prediction

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/predict
```

Response:

```json
{
    "class_id": 0,
    "confidence": 0.95,
    "processing_time": 0.1,
    "gpu_utilization": 0.75
}
```

## Model Details

- Architecture: ResNet50-based classifier
- Training Dataset: ImageNet
- Input: RGB images (224x224 pixels)
- Output: 1000 class probabilities
- Performance: < 3 seconds inference time
- Format: ONNX for optimized GPU inference
- GPU Acceleration: CUDA-enabled ONNX Runtime

## Development

### Adding New Tests

1. Add test cases to `test.py` or `test_server.py`
2. Follow existing test patterns
3. Run tests to verify changes

### Modifying the API

1. Edit `app.py` for API changes
2. Update tests in both test files
3. Rebuild Docker image to apply changes

## Troubleshooting

### Common Issues

1. **Docker Build Fails**

   - Ensure Docker is running
   - Check internet connection for package downloads
   - Verify Dockerfile syntax
   - Confirm NVIDIA Container Toolkit is installed
2. **Model Loading Errors**

   - Check if model.onnx exists
   - Verify model file permissions
   - Ensure correct Python version
   - Verify GPU availability
3. **API Connection Issues**

   - Verify port 8000 is not in use
   - Check container is running
   - Ensure correct API endpoint format
4. **Cerebrium Deployment Issues**

   - Verify API key configuration
   - Check model file size limits
   - Monitor deployment logs
   - Confirm GPU availability in Cerebrium
5. **GPU-related Issues**

   - Check NVIDIA drivers are installed
   - Verify CUDA version compatibility
   - Monitor GPU memory usage
   - Check GPU temperature and utilization

### Logs

- Container logs: `docker logs <container-id>`
- Application logs: Check stdout/stderr in container
- Cerebrium logs: Available in dashboard
- GPU metrics: NVIDIA-SMI output

## Performance Considerations

- Model inference time should be < 3 seconds
- Memory usage is optimized for container deployment
- Batch processing is supported for multiple images
- Monitoring metrics available in Cerebrium dashboard
- GPU utilization and memory monitoring
- CUDA performance optimization

## Security Notes

- No hardcoded credentials in the code
- Input validation for all API endpoints
- Error handling for malformed requests
- Secure API key management for Cerebrium
- GPU resource isolation