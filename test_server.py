import argparse
import requests
import time
import sys
import json
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CerebriumTester:
    def __init__(self, api_key: str, endpoint_url: str):
        """Initialize Cerebrium tester with API credentials.
        
        Args:
            api_key: Cerebrium API key
            endpoint_url: Full URL of the deployed model endpoint
        """
        self.api_key = api_key
        self.endpoint_url = endpoint_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
        self.test_results: List[Dict] = []

    def predict_image(self, image_path: str) -> Dict:
        """Send prediction request for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing prediction results
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        try:
            with open(image_path, "rb") as f:
                files = {"file": ("image.jpg", f, "image/jpeg")}
                response = requests.post(
                    f"{self.endpoint_url}/predict",
                    headers=self.headers,
                    files=files
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            raise

    def test_health(self) -> Dict:
        """Test the health endpoint.
        
        Returns:
            Dict containing health check results
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        try:
            response = requests.get(
                f"{self.endpoint_url}/health",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise

    def run_performance_test(
        self,
        image_path: str,
        num_requests: int = 10,
        concurrent: bool = False
    ) -> Dict:
        """Test model performance with multiple requests.
        
        Args:
            image_path: Path to the test image
            num_requests: Number of requests to make
            concurrent: Whether to make concurrent requests
            
        Returns:
            Dict containing performance metrics
        """
        latencies = []
        errors = 0
        
        if concurrent:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_requests
            ) as executor:
                futures = [
                    executor.submit(self.predict_image, image_path)
                    for _ in range(num_requests)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        start_time = time.time()
                        future.result()
                        latencies.append(time.time() - start_time)
                    except Exception as e:
                        errors += 1
                        logger.error(
                            f"Concurrent request failed: {str(e)}"
                        )
        else:
            for _ in range(num_requests):
                try:
                    start_time = time.time()
                    self.predict_image(image_path)
                    latencies.append(time.time() - start_time)
                except Exception as e:
                    errors += 1
                    logger.error(f"Request failed: {str(e)}")
        
        if not latencies:
            return {
                "status": "failed",
                "error_rate": 1.0,
                "message": "All requests failed"
            }
        
        return {
            "status": "success",
            "average_latency": sum(latencies) / len(latencies),
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
            "p99_latency": sorted(latencies)[int(0.99 * len(latencies))],
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "error_rate": errors / num_requests,
            "num_requests": num_requests,
            "successful_requests": len(latencies)
        }

    def run_custom_tests(self, test_cases: List[Tuple[str, int]]) -> List[Dict]:
        """Run custom test cases against the deployed model.
        
        Args:
            test_cases: List of (image_path, expected_class) tuples
            
        Returns:
            List of test results
        """
        results = []
        for image_path, expected_class in test_cases:
            try:
                result = self.predict_image(image_path)
                success = result["class_id"] == expected_class
                results.append({
                    "image": image_path,
                    "expected_class": expected_class,
                    "predicted_class": result["class_id"],
                    "confidence": result["confidence"],
                    "processing_time": result["processing_time"],
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(
                    f"Test {image_path}: "
                    f"Expected {expected_class}, "
                    f"got {result['class_id']}, "
                    f"Confidence: {result['confidence']:.4f}"
                )
            except Exception as e:
                results.append({
                    "image": image_path,
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                })
                logger.error(f"Test failed for {image_path}: {str(e)}")
        
        self.test_results.extend(results)
        return results

    def monitor_model(self, interval: int = 300, duration: int = 3600) -> None:
        """Monitor model performance over time.
        
        Args:
            interval: Time between checks in seconds
            duration: Total monitoring duration in seconds
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                health = self.test_health()
                logger.info(f"Health check: {json.dumps(health, indent=2)}")
                
                # Add custom monitoring metrics here
                # For example: memory usage, GPU utilization, etc.
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring check failed: {str(e)}")
                time.sleep(interval)

    def save_test_results(self, output_file: str) -> None:
        """Save test results to a JSON file.
        
        Args:
            output_file: Path to save results
        """
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Test results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Cerebrium model deployment"
    )
    parser.add_argument("--api-key", required=True, help="Cerebrium API key")
    parser.add_argument("--endpoint", required=True, help="Model endpoint URL")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run preset tests"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable monitoring"
    )
    parser.add_argument(
        "--output",
        default="test_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    tester = CerebriumTester(args.api_key, args.endpoint)
    
    try:
        # Test health endpoint
        health = tester.test_health()
        logger.info(f"Health check: {json.dumps(health, indent=2)}")
        
        if args.test_mode:
            # Run preset tests
            test_cases = [
                ("n01440764_tench.jpeg", 0),
                ("n01667114_mud_turtle.JPEG", 35)
            ]
            tester.run_custom_tests(test_cases)
            
            # Run performance tests
            perf_results = tester.run_performance_test(
                test_cases[0][0],
                num_requests=10,
                concurrent=True
            )
            logger.info(
                f"Performance results: {json.dumps(perf_results, indent=2)}"
            )
            
            # Save results
            tester.save_test_results(args.output)
        
        elif args.image:
            # Single image prediction
            result = tester.predict_image(args.image)
            logger.info(f"Prediction result: {json.dumps(result, indent=2)}")
        
        if args.monitor:
            # Start monitoring
            logger.info("Starting model monitoring...")
            tester.monitor_model()
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()