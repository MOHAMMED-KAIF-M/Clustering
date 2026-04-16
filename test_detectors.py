#!/usr/bin/env python3
"""
Quick test script for enhanced detection
Run this to see which detector works best for your images
"""

import sys
import logging
from pathlib import Path
import json
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_detector(image_path, detector_type):
    """Test a single detector"""
    try:
        from enhanced_detector import detect_image
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {detector_type.upper()} on {Path(image_path).name}")
        logger.info(f"{'='*60}")
        
        detections = detect_image(image_path, detector_type=detector_type)
        
        if not detections:
            logger.warning(f"No detections found with {detector_type}")
            return []
        
        # Prepare results table
        results = []
        for det in detections[:15]:  # Show top 15
            results.append([
                det.class_name,
                f"{det.confidence:.1%}",
                f"{det.bbox[0]:.0f}, {det.bbox[1]:.0f}"
            ])
        
        print(tabulate(results, 
                      headers=["Object", "Confidence", "Position (x,y)"],
                      tablefmt="grid"))
        
        logger.info(f"✓ Total detections: {len(detections)}")
        return detections
        
    except Exception as e:
        logger.error(f"✗ Error with {detector_type}: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_detectors.py <image_path> [detector_type]")
        print("\nExamples:")
        print("  python test_detectors.py input/photo1.jpg")
        print("  python test_detectors.py input/photo1.jpg yolo")
        print("  python test_detectors.py input/photo1.jpg claude")
        print("  python test_detectors.py input/photo1.jpg ollama")
        print("\nAll detectors:")
        print("  python test_detectors.py input/photo1.jpg all")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    detector_type = sys.argv[2] if len(sys.argv) > 2 else "yolo"
    
    # Check for API key if using Claude
    if detector_type == "claude":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("⚠️  Claude requires ANTHROPIC_API_KEY environment variable")
            print("   Get free API key: https://console.anthropic.com/")
            print("   Then run: export ANTHROPIC_API_KEY='sk-ant-...'")
    
    # Test single detector
    if detector_type != "all":
        test_detector(image_path, detector_type)
    else:
        # Test all available detectors
        print("\n🔍 TESTING ALL DETECTORS\n")
        
        detectors = ["yolo", "claude", "ollama"]
        results = {}
        
        for det_type in detectors:
            detections = test_detector(image_path, det_type)
            results[det_type] = {
                "count": len(detections),
                "objects": [d.class_name for d in detections[:10]]
            }
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        summary_table = []
        for det_type, data in results.items():
            summary_table.append([
                det_type.upper(),
                data["count"],
                ", ".join(data["objects"][:3]) + "..." if data["objects"] else "None"
            ])
        
        print(tabulate(summary_table,
                      headers=["Detector", "Total Objects", "Sample Objects"],
                      tablefmt="grid"))


if __name__ == "__main__":
    main()
