"""
Enhanced Object Detection Module
Supports multiple detection backends for maximum accuracy:
1. Grounding DINO - Open vocabulary detection
2. YOLOv8 Large - Better than nano for diverse objects
3. Claude Vision API - Multimodal understanding
4. Ollama Vision Models - Free alternative
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import base64
import urllib.request
import urllib.error

import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result"""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    segmentation: Optional[list] = None
    
    def to_dict(self):
        return asdict(self)


class YOLOv8LargeDetector:
    """YOLOv8 Large model - much better than nano"""
    
    def __init__(self, model_size: str = "l", device: str = "cuda"):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            device: 'cuda' or 'cpu'
        """
        self.model_size = model_size
        self.device = device
        model_name = f"yolov8{model_size}-seg.pt"  # Segmentation model
        
        logger.info(f"Loading YOLOv8{model_size.upper()} model...")
        self.model = YOLO(model_name)
        self.model.to(device)
        logger.info(f"YOLOv8{model_size.upper()} loaded successfully")
    
    def detect(self, image_path: str | Path, confidence: float = 0.25) -> list[Detection]:
        """Detect all objects in image"""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        results = self.model(image, conf=confidence, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                coords = box.xyxy[0]
                bbox = tuple(float(c) for c in coords)
                
                # Get segmentation mask if available
                segmentation = None
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[result.boxes == box][0]
                    segmentation = mask.cpu().numpy().tolist() if torch.is_tensor(mask) else mask.tolist()
                
                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    segmentation=segmentation
                ))
        
        logger.info(f"Detected {len(detections)} objects in {image_path}")
        return detections


class GroundingDINODetector:
    """Grounding DINO - Open vocabulary detection with text prompts"""
    
    def __init__(self, device: str = "cuda"):
        """Initialize Grounding DINO"""
        self.device = device
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            from groundingdino.util import box_ops
            
            self.load_model = load_model
            self.load_image = load_image
            self.predict = predict
            self.box_ops = box_ops
            
            config_path = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
            ckpt_path = "groundingdino/weights/groundingdino_swinb_cogvlm.pth"
            
            logger.info("Loading Grounding DINO model...")
            self.model = self.load_model(config_path, ckpt_path, device=device)
            logger.info("Grounding DINO loaded successfully")
        except ImportError as e:
            logger.warning(f"Grounding DINO not available: {e}")
            self.model = None
    
    def detect(self, image_path: str | Path, text_prompt: str = "", confidence: float = 0.3) -> list[Detection]:
        """Detect objects using text prompts"""
        if self.model is None:
            logger.warning("Grounding DINO model not available")
            return []
        
        # Comprehensive prompt for detecting all objects
        if not text_prompt:
            text_prompt = (
                "roof. window. building facade. door. glass door. tree. balcony. "
                "railing. wall. floor. sky. grass. cloud. outdoor. tree. plants. "
                "fence. patio. driveway. pool. furniture. chair. table. bed. sofa"
            )
        
        try:
            image_source, image = self.load_image(str(image_path))
            
            boxes, logits, phrases = self.predict(
                self.model,
                image,
                text_prompt,
                box_threshold=confidence,
                text_threshold=0.25
            )
            
            detections = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                # Convert from [0,1] to pixel coordinates
                h, w = image_source.shape[:2]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                
                detections.append(Detection(
                    class_name=phrase,
                    confidence=float(logit),
                    bbox=(x1, y1, x2, y2)
                ))
            
            logger.info(f"Detected {len(detections)} objects with Grounding DINO")
            return detections
        except Exception as e:
            logger.error(f"Grounding DINO detection failed: {e}")
            return []


class OllamaVisionDetector:
    """Ollama Vision Model - Free local alternative"""
    
    def __init__(self, model: str = "llava:latest", ollama_host: str = "http://localhost:11434"):
        """
        Args:
            model: Ollama model to use (llava, bakllava, etc)
            ollama_host: Ollama server URL
        """
        self.model = model
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/generate"
    
    def detect(self, image_path: str | Path) -> list[Detection]:
        """Detect objects using Ollama vision model"""
        try:
            image_path = Path(image_path)
            
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Prepare prompt for comprehensive detection
            prompt = """List ALL objects visible in this image. For each object, provide:
1. Object name
2. Approximate location (top, bottom, left, right, center)
3. Confidence (high, medium, low)

Include everything: furniture, architecture, nature elements, sky, walls, floors, textures, etc.
Format as JSON array with {name, location, confidence} for each object."""
            
            # Call Ollama API
            data = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
            
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"Querying {self.model} for object detection...")
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_text = result.get("response", "")
            
            # Parse response and extract detections
            detections = self._parse_ollama_response(response_text)
            logger.info(f"Detected {len(detections)} objects with Ollama")
            return detections
            
        except Exception as e:
            logger.error(f"Ollama detection failed: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return []
    
    def _parse_ollama_response(self, response_text: str) -> list[Detection]:
        """Parse Ollama response to extract detections"""
        detections = []
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                objects = json.loads(json_match.group())
                for obj in objects:
                    detections.append(Detection(
                        class_name=obj.get("name", "unknown"),
                        confidence=0.7 if obj.get("confidence") == "high" else 0.5,
                        bbox=(0, 0, 100, 100)  # Ollama doesn't return bbox
                    ))
        except Exception as e:
            logger.warning(f"Could not parse Ollama response: {e}")
        
        return detections


class ClaudeVisionDetector:
    """Claude Vision API - Highest accuracy for complex scenes"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def detect(self, image_path: str | Path) -> list[Detection]:
        """Detect objects using Claude Vision"""
        try:
            image_path = Path(image_path)
            
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            # Determine media type
            suffix = image_path.suffix.lower()
            media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")
            
            # Call Claude Vision
            prompt = """Analyze this image and list ALL visible objects and elements. Include:
- Architectural elements (walls, windows, doors, roof, facade, etc)
- Furniture and indoor items
- Natural elements (trees, sky, grass, clouds, plants, water)
- Textures and materials visible
- Any other notable objects or features

For each object/element, estimate:
1. What it is (precise name)
2. Approximate location in image (top, bottom, left, right, center, etc)
3. How clearly visible (high/medium/low confidence)

Format as JSON array with this structure:
[
  {"name": "object name", "location": "position description", "confidence": "high/medium/low"},
  ...
]

Be comprehensive - include everything visible, no matter how small."""
            
            logger.info("Querying Claude Vision for object detection...")
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            response_text = message.content[0].text
            detections = self._parse_claude_response(response_text)
            logger.info(f"Detected {len(detections)} objects with Claude Vision")
            return detections
            
        except Exception as e:
            logger.error(f"Claude Vision detection failed: {e}")
            return []
    
    def _parse_claude_response(self, response_text: str) -> list[Detection]:
        """Parse Claude response to extract detections"""
        detections = []
        try:
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                objects = json.loads(json_match.group())
                for obj in objects:
                    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                    detections.append(Detection(
                        class_name=obj.get("name", "unknown"),
                        confidence=confidence_map.get(obj.get("confidence", "medium").lower(), 0.7),
                        bbox=(0, 0, 100, 100)  # Claude doesn't return bbox
                    ))
        except Exception as e:
            logger.warning(f"Could not parse Claude response: {e}")
        
        return detections


class HybridDetector:
    """Combines multiple detectors for best accuracy"""
    
    def __init__(self, use_yolo: bool = True, use_grounding_dino: bool = True, 
                 use_claude: bool = False, use_ollama: bool = False):
        """Initialize hybrid detector with selected backends"""
        self.detectors = {}
        
        if use_yolo:
            try:
                self.detectors['yolo'] = YOLOv8LargeDetector(model_size="l")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv8L: {e}")
        
        if use_grounding_dino:
            try:
                self.detectors['grounding_dino'] = GroundingDINODetector()
            except Exception as e:
                logger.warning(f"Failed to load Grounding DINO: {e}")
        
        if use_claude:
            try:
                self.detectors['claude'] = ClaudeVisionDetector()
            except Exception as e:
                logger.warning(f"Failed to load Claude Vision: {e}")
        
        if use_ollama:
            try:
                self.detectors['ollama'] = OllamaVisionDetector()
            except Exception as e:
                logger.warning(f"Failed to load Ollama Vision: {e}")
        
        if not self.detectors:
            logger.error("No detectors available!")
            raise RuntimeError("At least one detection backend must be available")
    
    def detect(self, image_path: str | Path, merge_results: bool = True) -> list[Detection]:
        """Detect using all available backends"""
        all_detections = {}
        
        for backend_name, detector in self.detectors.items():
            try:
                logger.info(f"Running {backend_name} detection...")
                detections = detector.detect(image_path)
                all_detections[backend_name] = detections
            except Exception as e:
                logger.error(f"Error in {backend_name}: {e}")
        
        if merge_results:
            return self._merge_detections(all_detections)
        return all_detections
    
    def _merge_detections(self, all_detections: dict) -> list[Detection]:
        """Merge detections from multiple sources"""
        merged = {}
        
        for backend_name, detections in all_detections.items():
            for det in detections:
                key = det.class_name.lower()
                if key not in merged:
                    merged[key] = {
                        'class_name': det.class_name,
                        'confidences': [],
                        'bbox': det.bbox
                    }
                merged[key]['confidences'].append(det.confidence)
        
        result = []
        for key, data in merged.items():
            avg_confidence = np.mean(data['confidences'])
            result.append(Detection(
                class_name=data['class_name'],
                confidence=min(avg_confidence, 0.99),  # Cap at 0.99
                bbox=data['bbox']
            ))
        
        return sorted(result, key=lambda x: x.confidence, reverse=True)


def detect_image(image_path: str | Path, detector_type: str = "hybrid", **kwargs) -> list[Detection]:
    """
    Detect all objects in an image
    
    Args:
        image_path: Path to image
        detector_type: 'yolo', 'grounding_dino', 'claude', 'ollama', or 'hybrid'
        **kwargs: Additional arguments for detector
    
    Returns:
        List of Detection objects
    """
    image_path = Path(image_path)
    
    if detector_type == "yolo":
        detector = YOLOv8LargeDetector(**kwargs)
    elif detector_type == "grounding_dino":
        detector = GroundingDINODetector(**kwargs)
    elif detector_type == "claude":
        detector = ClaudeVisionDetector(**kwargs)
    elif detector_type == "ollama":
        detector = OllamaVisionDetector(**kwargs)
    elif detector_type == "hybrid":
        detector = HybridDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    return detector.detect(image_path)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_detector.py <image_path> [detector_type] [--claude] [--ollama]")
        print("detector_type: yolo (default), grounding_dino, claude, ollama, hybrid")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detector_type = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    
    use_claude = "--claude" in sys.argv
    use_ollama = "--ollama" in sys.argv
    
    detections = detect_image(
        image_path, 
        detector_type=detector_type,
        use_claude=use_claude,
        use_ollama=use_ollama
    )
    
    print(f"\n{'='*60}")
    print(f"Detected {len(detections)} objects:")
    print(f"{'='*60}")
    for det in detections:
        print(f"  {det.class_name:<30} Confidence: {det.confidence:.2%} BBox: {det.bbox}")
