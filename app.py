from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from model import ONNXClassifier
import time
import logging

from schema import PredictionResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="""
    A FastAPI application for image classification using an ONNX model.
    
    ## Features
    * Image classification with confidence scores
    * Health monitoring
    
    ## Endpoints
    * `/predict` - Classify an uploaded image
    * `/health` - Check API health status
    
    ## Model Information
    * Input: RGB image (224x224 pixels)
    * Output: Class ID (0-999) and confidence score
    * Dataset: ImageNet
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize model
try:
    model = ONNXClassifier()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise RuntimeError(f"Failed to initialize model: {str(e)}")


def validate_image(file: UploadFile) -> None:
    """
    Validate uploaded image file.
    
    Args:
        file: The uploaded file to validate
        
    Raises:
        HTTPException: If file is not an image or exceeds size limit
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Check file size (max 10MB)
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size must be less than 10MB"
            )
    file.file.seek(0)  # Reset file pointer


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify an image",
    description="""
    Classify an uploaded image using the ONNX model.
    
    ## Request
    * **Content-Type**: multipart/form-data
    * **File**: Image file (JPEG, PNG, etc.)
    * **Max Size**: 10MB
    
    ## Response
    * **class_id**: Predicted class ID (0-999)
    * **confidence**: Confidence score (0-1)
    * **processing_time**: Processing time in seconds
    
    ## Error Responses
    * **400 Bad Request**: Invalid file type or size
    * **500 Internal Server Error**: Model prediction failed
    
    ## Example
    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@image.jpg"
    ```
    """,
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "class_id": 0,
                        "confidence": 0.9876,
                        "processing_time": 0.123
                    }
                }
            }
        },
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "File must be an image"
                    }
                }
            }
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Prediction failed: [error details]"
                    }
                }
            }
        }
    }
)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image.
    
    Args:
        file: The image file to classify
        
    Returns:
        PredictionResponse: The prediction results
        
    Raises:
        HTTPException: If prediction fails or input is invalid
    """
    try:
        # Validate input
        validate_image(file)
        
        # Process image
        start_time = time.time()
        image_bytes = await file.read()
        # predictions, class_id = model.predict(image_bytes)
        predictions = model.predict(image_bytes)
        class_id = int(predictions.argmax())
        processing_time = time.time() - start_time
        
        image_bytes = await file.read()
        
        # Get confidence score
        confidence = float(predictions.max())
        
        logger.info(
            f"Prediction successful - Class: {class_id}, "
            f"Confidence: {confidence:.4f}, "
            f"Time: {processing_time:.3f}s"
        )
        
        return PredictionResponse(
            class_id=int(class_id),
            confidence=confidence,
            processing_time=processing_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API health",
    description="""
    Check the health status of the API and model.
    
    ## Response
    * **status**: Current API status
    * **model_loaded**: Whether the model is loaded
    * **version**: API version
    
    ## Example
    ```bash
    curl -X GET "http://localhost:8000/health" \
         -H "accept: application/json"
    ```
    """,
    responses={
        200: {
            "description": "Health check successful",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model_loaded": True,
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: The health status information
    """
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0"
    )