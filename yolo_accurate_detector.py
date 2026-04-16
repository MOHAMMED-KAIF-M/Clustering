"""
Optimized YOLOv8 Detector for Maximum Accuracy
Uses largest model (YOLOv8 Extra Large) for best detection
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result"""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    area: float  # Pixel area of bounding box
    
    def to_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": self.bbox,
            "area": float(self.area)
        }


class YOLOv11AccurateDetector:
    """
    YOLOv11 with maximum accuracy settings
    Uses largest model: YOLOv11x (Extra Large)
    """
    
    def __init__(self, model_size: str = "x", device: str = "cuda", confidence: float = 0.25):
        """
        Args:
            model_size: 'n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=extra large
            device: 'cuda' or 'cpu'
            confidence: Detection confidence threshold (lower = more detections)
        """
        self.model_size = model_size
        self.device = device
        self.confidence = confidence
        
        model_name = f"yolov11{model_size}-seg.pt"  # Segmentation model for masks (YOLOv11)
        
        logger.info(f"🚀 Loading YOLOv11{model_size.upper()} model (High Accuracy)...")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Confidence threshold: {confidence}")
        
        self.model = YOLO(model_name)
        self.model.to(device)
        
        logger.info(f"✓ YOLOv11{model_size.upper()} loaded successfully")
        logger.info(f"  Classes: {len(self.model.names)}")
    
    def detect(self, image_path: str | Path, confidence: Optional[float] = None) -> list[Detection]:
        """
        Detect all objects in image with high accuracy
        
        Args:
            image_path: Path to image
            confidence: Override default confidence threshold
        
        Returns:
            List of Detection objects sorted by confidence
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"❌ Image not found: {image_path}")
            return []
        
        # Use provided confidence or default
        conf = confidence if confidence is not None else self.confidence
        
        logger.info(f"🔍 Processing: {image_path.name}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"❌ Failed to load image: {image_path}")
            return []
        
        image_h, image_w = image.shape[:2]
        logger.info(f"   Image size: {image_w}x{image_h}")
        
        # Run detection with high accuracy settings
        results = self.model(
            image,
            conf=conf,  # Confidence threshold
            iou=0.45,   # IoU threshold (lower = more detections)
            verbose=False,
            augment=True  # TTA (Test Time Augmentation) for higher accuracy
        )
        
        detections = []
        
        for result in results:
            # Extract detections
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence_score = float(box.conf[0])
                
                # Get bounding box coordinates
                coords = box.xyxy[0]
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                
                # Calculate area (larger detections are more reliable)
                area = (x2 - x1) * (y2 - y1)
                
                detection = Detection(
                    class_name=class_name,
                    confidence=confidence_score,
                    bbox=(x1, y1, x2, y2),
                    area=area
                )
                detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"✓ Detected {len(detections)} objects")
        
        return detections
    
    def detect_and_visualize(self, image_path: str | Path, output_path: Optional[str | Path] = None, 
                           confidence: Optional[float] = None, show_confidence: bool = True) -> list[Detection]:
        """
        Detect objects and create visualization with bounding boxes
        
        Args:
            image_path: Input image path
            output_path: Where to save visualization (optional)
            confidence: Detection confidence threshold
            show_confidence: Whether to show confidence scores
        
        Returns:
            List of detections
        """
        detections = self.detect(image_path, confidence)
        
        if not detections:
            logger.warning("No detections found")
            return detections
        
        # Load image for visualization
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Colors for different confidence levels
        colors = {
            'high': (0, 255, 0),      # Green
            'medium': (255, 255, 0),  # Yellow
            'low': (255, 165, 0)      # Orange
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Determine color based on confidence
            if det.confidence >= 0.8:
                color = colors['high']
            elif det.confidence >= 0.6:
                color = colors['medium']
            else:
                color = colors['low']
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{det.class_name}"
            if show_confidence:
                label += f" {det.confidence:.0%}"
            
            # Draw label background
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
        
        # Save visualization
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"✓ Saved visualization: {output_path}")
        
        return detections


def detect_image(image_path: str | Path, model_size: str = "x", 
                confidence: float = 0.25, device: str = "cuda",
                visualize: bool = False, output_path: Optional[str | Path] = None) -> list[Detection]:
    """
    Simple function to detect objects in an image
    
    Args:
        image_path: Path to image
        model_size: 'n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=extra large
        confidence: Detection confidence threshold (0.0-1.0, lower = more detections)
        device: 'cuda' or 'cpu'
        visualize: Whether to create visualization
        output_path: Path to save visualization
    
    Returns:
        List of Detection objects
    """
    detector = YOLOv11AccurateDetector(model_size=model_size, device=device, confidence=confidence)
    
    if visualize:
        return detector.detect_and_visualize(image_path, output_path, confidence)
    else:
        return detector.detect(image_path, confidence)


def batch_detect(image_dir: str | Path, model_size: str = "x", 
                confidence: float = 0.25, output_dir: Optional[str | Path] = None) -> dict:
    """
    Detect objects in all images in a directory
    
    Args:
        image_dir: Directory containing images
        model_size: YOLO model size
        confidence: Detection threshold
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary mapping image names to detections
    """
    image_dir = Path(image_dir)
    detector = YOLOv11AccurateDetector(model_size=model_size, confidence=confidence)
    
    results = {}
    
    for image_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        logger.info(f"\n{'='*60}")
        detections = detector.detect(image_path)
        results[image_path.name] = [d.to_dict() for d in detections]
        
        if output_dir:
            output_path = Path(output_dir) / f"{image_path.stem}_detected.jpg"
            detector.detect_and_visualize(image_path, output_path, confidence)
    
    return results


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("🎯 YOLOv11 High-Accuracy Object Detector")
        print("=" * 70)
        print("\nUsage:")
        print("  python yolo_accurate_detector.py <image_path> [--model n|s|m|l|x] [--conf 0.25]")
        print("\nExamples:")
        print("  python yolo_accurate_detector.py input/photo.jpg")
        print("  python yolo_accurate_detector.py input/photo.jpg --model x --conf 0.2")
        print("  python yolo_accurate_detector.py input/photo.jpg --visualize --output output/")
        print("\nModel sizes:")
        print("  n = nano (fast, less accurate)")
        print("  s = small")
        print("  m = medium")
        print("  l = large")
        print("  x = extra large (most accurate, slower)")
        print("\nConfidence threshold (default 0.25):")
        print("  Lower = more detections (more sensitive)")
        print("  Higher = fewer detections (more precise)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Parse arguments
    model_size = "x"  # Default to largest
    confidence = 0.25  # Default
    visualize = False
    output_path = None
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_size = sys.argv[i + 1]
        elif arg == "--conf" and i + 1 < len(sys.argv):
            confidence = float(sys.argv[i + 1])
        elif arg == "--visualize":
            visualize = True
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
    
    print("\n" + "=" * 70)
    print("🎯 YOLOv11 High-Accuracy Detection")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Model: YOLOv11{model_size.upper()}")
    print(f"Confidence threshold: {confidence}")
    print("=" * 70 + "\n")
    
    # Run detection
    detections = detect_image(image_path, model_size=model_size, confidence=confidence, 
                            visualize=visualize, output_path=output_path)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS - Detected {len(detections)} Objects")
    print(f"{'='*70}")
    
    if detections:
        print(f"{'Object':<30} {'Confidence':<15} {'Position (x,y)':<20} {'Size':<15}")
        print("-" * 70)
        
        for i, det in enumerate(detections[:20], 1):  # Show top 20
            x1, y1, x2, y2 = det.bbox
            width = int(x2 - x1)
            height = int(y2 - y1)
            print(f"{det.class_name:<30} {det.confidence:>6.1%}{'':8} ({int(x1):4.0f}, {int(y1):4.0f})  {width:4}x{height:<4}")
        
        if len(detections) > 20:
            print(f"... and {len(detections) - 20} more objects")
    else:
        print("No objects detected")
    
    print(f"\n{'='*70}\n")
