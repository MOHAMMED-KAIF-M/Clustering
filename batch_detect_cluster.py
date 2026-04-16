"""
Batch Detection with Clustering
Reads images from input folder, detects objects, and clusters by detected objects
Saves results to new_outputs folder
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN

from yolo_accurate_detector import YOLOv8AccurateDetector, Detection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectClusterer:
    """Groups images by detected objects"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: How similar object sets should be to cluster (0-1)
        """
        self.similarity_threshold = similarity_threshold
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def cluster_by_objects(self, image_detections: dict[str, list[Detection]]) -> dict[int, list[str]]:
        """
        Cluster images based on similarity of detected objects
        
        Args:
            image_detections: {image_name: [detections]}
        
        Returns:
            {cluster_id: [image_names]}
        """
        if not image_detections:
            return {}
        
        # Convert detections to object sets
        image_names = list(image_detections.keys())
        object_sets = [set(d.class_name for d in image_detections[name]) for name in image_names]
        
        # Build similarity matrix
        n = len(image_names)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = self.jaccard_similarity(object_sets[i], object_sets[j])
        
        # Convert to distance matrix
        distance_matrix = 1.0 - similarity_matrix
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=1.0 - self.similarity_threshold, min_samples=1, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        # Group images by cluster
        clusters = defaultdict(list)
        for image_name, label in zip(image_names, labels):
            clusters[int(label)].append(image_name)
        
        return dict(clusters)


class BatchDetector:
    """Process multiple images with detection and clustering"""
    
    def __init__(self, input_dir: str | Path, output_dir: str | Path, model_size: str = "x", confidence: float = 0.25):
        """
        Args:
            input_dir: Directory with input images
            output_dir: Directory to save results
            model_size: YOLO model size
            confidence: Detection confidence threshold
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_size = model_size
        self.confidence = confidence
        
        self.detector = YOLOv8AccurateDetector(model_size=model_size, confidence=confidence)
        self.clusterer = ObjectClusterer()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all_images(self) -> dict:
        """Process all images in input folder"""
        
        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png"}
        image_paths = [p for p in self.input_dir.iterdir() 
                      if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            logger.error(f"❌ No images found in {self.input_dir}")
            return {}
        
        logger.info(f"📁 Found {len(image_paths)} images to process")
        logger.info(f"{'='*70}\n")
        
        # Detect objects in all images
        all_detections = {}
        detection_summary = defaultdict(int)
        
        for idx, image_path in enumerate(sorted(image_paths), 1):
            logger.info(f"[{idx}/{len(image_paths)}] Processing {image_path.name}")
            
            try:
                detections = self.detector.detect(image_path, self.confidence)
                all_detections[image_path.name] = detections
                
                # Count object types
                for det in detections:
                    detection_summary[det.class_name] += 1
                
                logger.info(f"    ✓ Detected {len(detections)} objects")
                
            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Detection complete: {len(all_detections)} images processed")
        logger.info(f"{'='*70}\n")
        
        return {
            "total_images": len(all_detections),
            "detections": all_detections,
            "summary": dict(detection_summary)
        }
    
    def cluster_images(self, all_detections: dict) -> dict:
        """Cluster images based on detected objects"""
        
        logger.info(f"🔗 Clustering images by detected objects...")
        
        clusters = self.clusterer.cluster_by_objects(all_detections)
        
        logger.info(f"✓ Created {len(clusters)} clusters\n")
        
        for cluster_id, image_names in sorted(clusters.items()):
            logger.info(f"  Cluster {cluster_id}: {len(image_names)} images")
            
            # Show sample objects in this cluster
            all_objects = set()
            for image_name in image_names:
                objects = set(d.class_name for d in all_detections[image_name])
                all_objects.update(objects)
            
            object_list = ", ".join(sorted(list(all_objects))[:5])
            if len(all_objects) > 5:
                object_list += f", ... and {len(all_objects) - 5} more"
            
            logger.info(f"    Objects: {object_list}\n")
        
        return clusters
    
    def save_results(self, all_detections: dict, clusters: dict):
        """Save detection results and create visualizations"""
        
        logger.info(f"💾 Saving results...\n")
        
        # Save global detection summary
        summary_data = {
            "total_images": len(all_detections),
            "total_clusters": len(clusters),
            "detections_by_image": {},
            "all_object_types": {}
        }
        
        for image_name, detections in all_detections.items():
            summary_data["detections_by_image"][image_name] = [d.to_dict() for d in detections]
            for det in detections:
                summary_data["all_object_types"][det.class_name] = summary_data["all_object_types"].get(det.class_name, 0) + 1
        
        summary_file = self.output_dir / "detection_summary.json"
        summary_file.write_text(json.dumps(summary_data, indent=2))
        logger.info(f"  ✓ Saved: {summary_file.name}")
        
        # Create cluster folders and save images
        for cluster_id, image_names in clusters.items():
            cluster_dir = self.output_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            # Get all objects in this cluster
            all_objects = set()
            for image_name in image_names:
                objects = set(d.class_name for d in all_detections[image_name])
                all_objects.update(objects)
            
            # Save cluster metadata
            cluster_data = {
                "cluster_id": cluster_id,
                "total_images": len(image_names),
                "objects_in_cluster": sorted(list(all_objects)),
                "images": image_names
            }
            
            cluster_meta_file = cluster_dir / "metadata.json"
            cluster_meta_file.write_text(json.dumps(cluster_data, indent=2))
            
            # Copy and annotate images
            for image_name in image_names:
                src_path = self.input_dir / image_name
                dst_path = cluster_dir / image_name
                
                if src_path.exists():
                    # Create annotated version
                    self._create_annotated_image(src_path, dst_path, all_detections[image_name])
            
            logger.info(f"  ✓ Created cluster_{cluster_id}: {len(image_names)} images, {len(all_objects)} object types")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Results saved to: {self.output_dir}")
        logger.info(f"{'='*70}\n")
    
    def _create_annotated_image(self, src_path: Path, dst_path: Path, detections: list[Detection]):
        """Create annotated image with bounding boxes"""
        try:
            image = Image.open(src_path)
            draw = ImageDraw.Draw(image)
            
            # Load font
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # Color mapping
            colors = {
                'high': (0, 255, 0),      # Green
                'medium': (255, 255, 0),  # Yellow
                'low': (255, 165, 0)      # Orange
            }
            
            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                
                # Determine color
                if det.confidence >= 0.8:
                    color = colors['high']
                elif det.confidence >= 0.6:
                    color = colors['medium']
                else:
                    color = colors['low']
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw label
                label = f"{det.class_name} {det.confidence:.0%}"
                bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
            
            image.save(dst_path)
        except Exception as e:
            logger.warning(f"Could not create annotated image for {src_path.name}: {e}")
            # Just copy original
            import shutil
            shutil.copy2(src_path, dst_path)
    
    def generate_report(self, all_detections: dict, clusters: dict):
        """Generate HTML report"""
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection & Clustering Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .summary h2 { color: #2196F3; }
        .stat { display: inline-block; margin: 10px 20px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .stat-label { font-size: 12px; color: #666; }
        .cluster { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .cluster h3 { color: #4CAF50; }
        .object-list { display: flex; flex-wrap: wrap; gap: 10px; }
        .object-tag { background: #E3F2FD; padding: 8px 12px; border-radius: 20px; font-size: 12px; }
        .image-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; }
        .image-item { background: #f9f9f9; padding: 10px; border-radius: 4px; text-align: center; font-size: 12px; }
    </style>
</head>
<body>
    <h1>🎯 Object Detection & Clustering Report</h1>
"""
        
        # Summary
        total_objects = sum(1 for dets in all_detections.values() for _ in dets)
        unique_objects = set()
        for dets in all_detections.values():
            for det in dets:
                unique_objects.add(det.class_name)
        
        html += f"""
    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        <div class="stat">
            <div class="stat-value">{len(all_detections)}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(clusters)}</div>
            <div class="stat-label">Clusters</div>
        </div>
        <div class="stat">
            <div class="stat-value">{total_objects}</div>
            <div class="stat-label">Total Objects Detected</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(unique_objects)}</div>
            <div class="stat-label">Unique Object Types</div>
        </div>
    </div>
"""
        
        # Top detected objects
        html += """
    <div class="summary">
        <h2>🏆 Top Detected Objects</h2>
        <div class="object-list">
"""
        
        object_counts = {}
        for dets in all_detections.values():
            for det in dets:
                object_counts[det.class_name] = object_counts.get(det.class_name, 0) + 1
        
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            html += f'        <div class="object-tag">{obj} ({count})</div>\n'
        
        html += """
        </div>
    </div>
"""
        
        # Clusters
        for cluster_id, image_names in sorted(clusters.items()):
            all_objects = set()
            for image_name in image_names:
                objects = set(d.class_name for d in all_detections[image_name])
                all_objects.update(objects)
            
            html += f"""
    <div class="cluster">
        <h3>Cluster {cluster_id} ({len(image_names)} images)</h3>
        <strong>Objects:</strong>
        <div class="object-list">
"""
            
            for obj in sorted(all_objects):
                html += f'            <div class="object-tag">{obj}</div>\n'
            
            html += """
        </div>
        <strong>Images:</strong>
        <div class="image-list">
"""
            
            for image_name in image_names[:10]:  # Show first 10
                html += f'            <div class="image-item">{image_name}</div>\n'
            
            if len(image_names) > 10:
                html += f'            <div class="image-item">... and {len(image_names) - 10} more</div>\n'
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        report_file = self.output_dir / "report.html"
        report_file.write_text(html)
        logger.info(f"  ✓ Generated: report.html")


def main():
    """Main execution"""
    
    input_dir = Path("input")
    output_dir = Path("new_outputs")
    
    # Check input directory
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        return
    
    logger.info("\n" + "="*70)
    logger.info("🎯 BATCH DETECTION & CLUSTERING")
    logger.info("="*70 + "\n")
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}\n")
    
    # Initialize processor
    processor = BatchDetector(
        input_dir=input_dir,
        output_dir=output_dir,
        model_size="x",        # YOLOv11 Extra Large
        confidence=0.25        # Standard confidence
    )
    
    # Process all images
    results = processor.process_all_images()
    
    if not results["detections"]:
        logger.error("No images could be processed")
        return
    
    # Cluster images
    clusters = processor.cluster_images(results["detections"])
    
    # Save results
    processor.save_results(results["detections"], clusters)
    
    # Generate report
    processor.generate_report(results["detections"], clusters)
    
    logger.info("\n✅ PROCESSING COMPLETE!\n")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info(f"   - Cluster folders with annotated images")
    logger.info(f"   - detection_summary.json (all detections)")
    logger.info(f"   - report.html (visual report)")


if __name__ == "__main__":
    main()
