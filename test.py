import unittest
import time
import numpy as np
from PIL import Image
import io
from model import ONNXClassifier
from fastapi.testclient import TestClient
from app import app


class TestONNXClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and load model"""
        cls.model = ONNXClassifier()
        cls.test_images = {
            "tench": ("n01440764_tench.jpeg", 0),
            "turtle": ("n01667114_mud_turtle.JPEG", 35)
        }
        cls.client = TestClient(app)

    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNotNone(self.model.session)
        self.assertIsNotNone(self.model.input_name)
        self.assertIsNotNone(self.model.output_name)
        self.assertEqual(self.model.input_name, "input")
        self.assertEqual(self.model.output_name, "output")

    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        # Test with a sample image
        with open(self.test_images["tench"][0], "rb") as f:
            image_bytes = f.read()
        processed = self.model.preprocess_image(image_bytes)
        
        # Check shape and value range
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        self.assertTrue(
            np.all(processed >= -3) and np.all(processed <= 3)
        )
        
        # Test RGB conversion
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        self.assertEqual(image.mode, 'RGB')

    def test_prediction_shape(self):
        """Test prediction output shape"""
        with open(self.test_images["tench"][0], "rb") as f:
            image_bytes = f.read()
        predictions = self.model.predict(image_bytes)
        
        # Check output shape (batch_size, num_classes)
        self.assertEqual(predictions.shape, (1, 1000))
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions))
        probs = exp_preds / np.sum(exp_preds)
        
        # Check if probabilities sum to 1
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
        # Check if all probabilities are between 0 and 1
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))

    def test_known_predictions(self):
        """Test predictions for known images"""
        for image_name, (file_name, expected_class) in self.test_images.items():
            with open(file_name, "rb") as f:
                image_bytes = f.read()
            predictions = self.model.predict(image_bytes)
            predicted_class = np.argmax(predictions)
            confidence = float(predictions.max())
            
            self.assertEqual(
                predicted_class,
                expected_class,
                f"Wrong prediction for {image_name}. "
                f"Expected {expected_class}, got {predicted_class}"
            )
            self.assertGreater(
                confidence, 0.5,
                f"Low confidence for {image_name}: {confidence}"
            )

    def test_invalid_input(self):
        """Test error handling for invalid inputs"""
        # Test with invalid image data
        with self.assertRaises((ValueError, RuntimeError)):
            self.model.predict(b"invalid image data")
        
        # Test with empty image
        with self.assertRaises((ValueError, RuntimeError)):
            self.model.predict(b"")
        
        # Test with non-image file
        with self.assertRaises((ValueError, RuntimeError)):
            self.model.predict(b"not an image")

    def test_performance(self):
        """Test model performance"""
        with open(self.test_images["tench"][0], "rb") as f:
            image_bytes = f.read()
        
        # Test inference time
        start_time = time.time()
        self.model.predict(image_bytes)
        inference_time = time.time() - start_time
        
        # Model should respond within 2-3 seconds as per requirements
        self.assertLess(
            inference_time, 3.0,
            f"Inference too slow: {inference_time:.2f}s"
        )

    def test_api_endpoints(self):
        """Test API endpoints"""
        # Test health endpoint
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        health_data = response.json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertTrue(health_data["model_loaded"])
        
        # Test prediction endpoint with valid image
        with open(self.test_images["tench"][0], "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            response = self.client.post("/predict", files=files)
            self.assertEqual(response.status_code, 200)
            pred_data = response.json()
            self.assertIn("class_id", pred_data)
            self.assertIn("confidence", pred_data)
            self.assertIn("processing_time", pred_data)
            self.assertEqual(pred_data["class_id"], 0)
            self.assertGreater(pred_data["confidence"], 0.5)
            self.assertLess(pred_data["processing_time"], 3.0)
        
        # Test prediction endpoint with invalid file
        response = self.client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()