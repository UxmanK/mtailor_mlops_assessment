import onnxruntime as ort
import numpy as np
from PIL import Image
import io

class ONNXClassifier:
    def __init__(self, model_path="/app/model.onnx"):
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess_image(self, image_bytes):
        """Preprocess image bytes to model input format"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Resize and center crop
            image = image.resize((256, 256))  # Slightly larger for center crop
            width, height = image.size
            left = (width - 224)/2
            top = (height - 224)/2
            right = (width + 224)/2
            bottom = (height + 224)/2
            image = image.crop((left, top, right, bottom))
            
            # Convert to numpy array and normalize to [0,1]
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Change from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_bytes):
        """Run prediction on preprocessed image"""
        try:
            input_array = self.preprocess_image(image_bytes)
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_array}
            )
            return outputs[0]
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
