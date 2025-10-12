"""
Comprehensive Dataset Analyzer for YOLO OBB Datasets

This module provides extensible classes for analyzing training datasets,
computing quality metrics, and generating visualizations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from tqdm import tqdm


@dataclass
class DatasetMetrics:
    """Container for dataset quality metrics."""
    
    # Basic counts
    total_images: int = 0
    total_annotations: int = 0
    images_with_annotations: int = 0
    images_without_annotations: int = 0
    
    # Class distribution
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    
    # Image metrics
    image_dimensions: List[Tuple[int, int]] = field(default_factory=list)
    image_aspect_ratios: List[float] = field(default_factory=list)
    image_sizes_mb: List[float] = field(default_factory=list)
    
    # Annotation metrics per image
    annotations_per_image: List[int] = field(default_factory=list)
    
    # Bounding box metrics
    bbox_areas: List[float] = field(default_factory=list)
    bbox_widths: List[float] = field(default_factory=list)
    bbox_heights: List[float] = field(default_factory=list)
    bbox_aspect_ratios: List[float] = field(default_factory=list)
    bbox_angles: List[float] = field(default_factory=list)
    
    # Spatial distribution (center points normalized)
    bbox_centers_x: List[float] = field(default_factory=list)
    bbox_centers_y: List[float] = field(default_factory=list)
    
    # Coverage metrics
    image_coverage_ratios: List[float] = field(default_factory=list)
    
    # Overlap metrics
    bbox_overlaps: List[float] = field(default_factory=list)
    
    # Quality indicators
    small_objects_count: int = 0  # area < 1% of image
    medium_objects_count: int = 0  # 1% <= area < 10%
    large_objects_count: int = 0  # area >= 10%
    
    # Edge proximity (objects near image edges)
    objects_near_edge: int = 0
    edge_threshold: float = 0.05  # 5% from edge
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'total_images': self.total_images,
            'total_annotations': self.total_annotations,
            'images_with_annotations': self.images_with_annotations,
            'images_without_annotations': self.images_without_annotations,
            'class_counts': self.class_counts,
            'class_names': self.class_names,
            'image_dimensions_unique': list(set(self.image_dimensions)),
            'annotations_per_image_stats': self._compute_stats(self.annotations_per_image),
            'bbox_area_stats': self._compute_stats(self.bbox_areas),
            'bbox_width_stats': self._compute_stats(self.bbox_widths),
            'bbox_height_stats': self._compute_stats(self.bbox_heights),
            'bbox_aspect_ratio_stats': self._compute_stats(self.bbox_aspect_ratios),
            'image_aspect_ratio_stats': self._compute_stats(self.image_aspect_ratios),
            'image_size_mb_stats': self._compute_stats(self.image_sizes_mb),
            'image_coverage_stats': self._compute_stats(self.image_coverage_ratios),
            'small_objects_count': self.small_objects_count,
            'medium_objects_count': self.medium_objects_count,
            'large_objects_count': self.large_objects_count,
            'objects_near_edge': self.objects_near_edge,
        }
    
    @staticmethod
    def _compute_stats(data: List[float]) -> Dict[str, float]:
        """Compute statistical summary of a list of values."""
        if not data:
            return {}
        
        arr = np.array(data)
        return {
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }


class YOLOOBBDatasetAnalyzer:
    """
    Comprehensive analyzer for YOLO OBB format datasets.
    
    Supports both standalone analysis and interactive notebook usage.
    """
    
    def __init__(self, dataset_path: str, load_images: bool = True):
        """
        Initialize the dataset analyzer.
        
        Args:
            dataset_path: Path to the dataset root directory
            load_images: Whether to load image metadata (can be slow for large datasets)
        """
        self.dataset_path = Path(dataset_path)
        self.load_images = load_images
        self.config = None
        self.metrics = {}
        self.splits = ['train', 'val', 'test']
        
        # Load dataset configuration
        self._load_config()
        
    def _load_config(self):
        """Load YAML configuration file."""
        yaml_files = list(self.dataset_path.glob('*.yaml'))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML config found in {self.dataset_path}")
        
        with open(yaml_files[0], 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loaded config: {yaml_files[0].name}")
        print(f"Dataset: {self.config.get('info', {}).get('description', 'N/A')}")
        print(f"Classes: {self.config.get('names', {})}")
        
    def analyze_split(self, split: str = 'train') -> DatasetMetrics:
        """
        Analyze a specific data split (train/val/test).
        
        Args:
            split: Name of the split to analyze
            
        Returns:
            DatasetMetrics object containing all computed metrics
        """
        metrics = DatasetMetrics()
        metrics.class_names = self.config.get('names', {})
        
        # Determine paths
        if split == 'train':
            img_dir = self.dataset_path / self.config.get('train', f'images/{split}')
            label_dir = self.dataset_path / 'labels' / split
        elif split == 'val':
            img_dir = self.dataset_path / self.config.get('val', f'images/{split}')
            label_dir = self.dataset_path / 'labels' / split
        elif split == 'test':
            img_dir = self.dataset_path / self.config.get('test', f'images/{split}')
            label_dir = self.dataset_path / 'labels' / split
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not img_dir.exists():
            print(f"Warning: Image directory not found: {img_dir}")
            return metrics
        
        if not label_dir.exists():
            print(f"Warning: Label directory not found: {label_dir}")
            return metrics
        
        # Get all images
        image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        metrics.total_images = len(image_files)
        
        print(f"\nAnalyzing {split} split: {metrics.total_images} images")
        
        # Analyze each image and its annotations
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            label_path = label_dir / f"{img_path.stem}.txt"
            
            # Load image metadata
            if self.load_images:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        metrics.image_dimensions.append((width, height))
                        metrics.image_aspect_ratios.append(width / height)
                        
                    # File size in MB
                    size_mb = img_path.stat().st_size / (1024 * 1024)
                    metrics.image_sizes_mb.append(size_mb)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue
            else:
                # Use default dimensions if not loading images
                width, height = 1000, 1000
            
            # Parse annotations
            if label_path.exists():
                annotations = self._parse_label_file(label_path)
                
                if annotations:
                    metrics.images_with_annotations += 1
                    metrics.annotations_per_image.append(len(annotations))
                    metrics.total_annotations += len(annotations)
                    
                    # Analyze each annotation
                    image_total_coverage = 0.0
                    for ann in annotations:
                        class_id = ann['class_id']
                        class_name = metrics.class_names.get(class_id, f"class_{class_id}")
                        metrics.class_counts[class_name] = metrics.class_counts.get(class_name, 0) + 1
                        
                        # Extract bounding box metrics
                        bbox_metrics = self._compute_bbox_metrics(ann['points'], width, height)
                        
                        metrics.bbox_areas.append(bbox_metrics['area'])
                        metrics.bbox_widths.append(bbox_metrics['width'])
                        metrics.bbox_heights.append(bbox_metrics['height'])
                        metrics.bbox_aspect_ratios.append(bbox_metrics['aspect_ratio'])
                        metrics.bbox_angles.append(bbox_metrics['angle'])
                        metrics.bbox_centers_x.append(bbox_metrics['center_x'])
                        metrics.bbox_centers_y.append(bbox_metrics['center_y'])
                        
                        image_total_coverage += bbox_metrics['area']
                        
                        # Classify object size
                        if bbox_metrics['area'] < 0.01:
                            metrics.small_objects_count += 1
                        elif bbox_metrics['area'] < 0.1:
                            metrics.medium_objects_count += 1
                        else:
                            metrics.large_objects_count += 1
                        
                        # Check edge proximity
                        if self._is_near_edge(bbox_metrics['center_x'], bbox_metrics['center_y'], 
                                             metrics.edge_threshold):
                            metrics.objects_near_edge += 1
                    
                    # Store image coverage ratio
                    metrics.image_coverage_ratios.append(min(image_total_coverage, 1.0))
                    
                    # Compute overlaps if multiple objects
                    if len(annotations) > 1:
                        overlaps = self._compute_overlaps(annotations, width, height)
                        metrics.bbox_overlaps.extend(overlaps)
                else:
                    metrics.images_without_annotations += 1
                    metrics.annotations_per_image.append(0)
            else:
                metrics.images_without_annotations += 1
                metrics.annotations_per_image.append(0)
        
        return metrics
    
    def _parse_label_file(self, label_path: Path) -> List[Dict]:
        """Parse a YOLO OBB format label file."""
        annotations = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:  # class_id + 8 coordinates
                        class_id = int(parts[0])
                        points = [float(x) for x in parts[1:9]]
                        annotations.append({
                            'class_id': class_id,
                            'points': points
                        })
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
        
        return annotations
    
    def _compute_bbox_metrics(self, points: List[float], img_width: int, img_height: int) -> Dict:
        """
        Compute metrics for an oriented bounding box.
        
        Args:
            points: List of 8 values [x1, y1, x2, y2, x3, y3, x4, y4] (normalized)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Dictionary of bbox metrics
        """
        # Reshape points to (4, 2) array
        pts = np.array(points).reshape(4, 2)
        
        # Compute center
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        
        # Compute area using Shoelace formula
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
        
        # Compute width and height (approximate from bounding rectangle)
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        
        # Compute rotation angle
        dx = pts[1, 0] - pts[0, 0]
        dy = pts[1, 1] - pts[0, 1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'angle': angle,
        }
    
    def _is_near_edge(self, center_x: float, center_y: float, threshold: float) -> bool:
        """Check if a point is near the image edge."""
        return (center_x < threshold or center_x > 1 - threshold or
                center_y < threshold or center_y > 1 - threshold)
    
    def _compute_overlaps(self, annotations: List[Dict], img_width: int, img_height: int) -> List[float]:
        """Compute IoU overlaps between bounding boxes."""
        overlaps = []
        
        # For simplicity, compute axis-aligned box IoU
        boxes = []
        for ann in annotations:
            pts = np.array(ann['points']).reshape(4, 2)
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            boxes.append([min_x, min_y, max_x, max_y])
        
        # Compute pairwise IoU
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self._compute_iou(boxes[i], boxes[j])
                overlaps.append(iou)
        
        return overlaps
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two axis-aligned boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Compute union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def analyze_all_splits(self) -> Dict[str, DatasetMetrics]:
        """Analyze all available splits in the dataset."""
        results = {}
        
        for split in self.splits:
            try:
                metrics = self.analyze_split(split)
                if metrics.total_images > 0:
                    results[split] = metrics
                    self.metrics[split] = metrics
            except Exception as e:
                print(f"Error analyzing {split} split: {e}")
        
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive text report of the dataset analysis.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            The report as a string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nDataset: {self.dataset_path}")
        report_lines.append(f"Description: {self.config.get('info', {}).get('description', 'N/A')}")
        report_lines.append(f"Classes: {self.config.get('names', {})}")
        report_lines.append(f"Number of classes: {self.config.get('nc', 'N/A')}")
        
        # Analyze each split
        for split, metrics in self.metrics.items():
            report_lines.append(f"\n{'-' * 80}")
            report_lines.append(f"{split.upper()} SPLIT")
            report_lines.append(f"{'-' * 80}")
            
            # Basic statistics
            report_lines.append(f"\n[BASIC STATISTICS]")
            report_lines.append(f"  Total images: {metrics.total_images}")
            report_lines.append(f"  Images with annotations: {metrics.images_with_annotations}")
            report_lines.append(f"  Images without annotations: {metrics.images_without_annotations}")
            report_lines.append(f"  Total annotations: {metrics.total_annotations}")
            report_lines.append(f"  Average annotations per image: {np.mean(metrics.annotations_per_image):.2f}")
            
            # Class distribution
            report_lines.append(f"\n[CLASS DISTRIBUTION]")
            for class_name, count in sorted(metrics.class_counts.items()):
                percentage = (count / metrics.total_annotations * 100) if metrics.total_annotations > 0 else 0
                report_lines.append(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            # Image statistics
            if metrics.image_dimensions:
                report_lines.append(f"\n[IMAGE STATISTICS]")
                report_lines.append(f"  Unique dimensions: {len(set(metrics.image_dimensions))}")
                report_lines.append(f"  Aspect ratios - Mean: {np.mean(metrics.image_aspect_ratios):.2f}, "
                                  f"Std: {np.std(metrics.image_aspect_ratios):.2f}")
                report_lines.append(f"  File sizes (MB) - Mean: {np.mean(metrics.image_sizes_mb):.2f}, "
                                  f"Min: {np.min(metrics.image_sizes_mb):.2f}, "
                                  f"Max: {np.max(metrics.image_sizes_mb):.2f}")
            
            # Bounding box statistics
            if metrics.bbox_areas:
                report_lines.append(f"\n[BOUNDING BOX STATISTICS]")
                report_lines.append(f"  Area (normalized) - Mean: {np.mean(metrics.bbox_areas):.4f}, "
                                  f"Median: {np.median(metrics.bbox_areas):.4f}, "
                                  f"Std: {np.std(metrics.bbox_areas):.4f}")
                report_lines.append(f"  Width (normalized) - Mean: {np.mean(metrics.bbox_widths):.4f}, "
                                  f"Min: {np.min(metrics.bbox_widths):.4f}, "
                                  f"Max: {np.max(metrics.bbox_widths):.4f}")
                report_lines.append(f"  Height (normalized) - Mean: {np.mean(metrics.bbox_heights):.4f}, "
                                  f"Min: {np.min(metrics.bbox_heights):.4f}, "
                                  f"Max: {np.max(metrics.bbox_heights):.4f}")
                report_lines.append(f"  Aspect ratio - Mean: {np.mean(metrics.bbox_aspect_ratios):.2f}, "
                                  f"Median: {np.median(metrics.bbox_aspect_ratios):.2f}")
                report_lines.append(f"  Rotation angles - Mean: {np.mean(metrics.bbox_angles):.1f}°, "
                                  f"Std: {np.std(metrics.bbox_angles):.1f}°")
            
            # Object size distribution
            report_lines.append(f"\n[OBJECT SIZE DISTRIBUTION]")
            total_objs = metrics.small_objects_count + metrics.medium_objects_count + metrics.large_objects_count
            if total_objs > 0:
                report_lines.append(f"  Small objects (<1% of image): {metrics.small_objects_count} "
                                  f"({metrics.small_objects_count/total_objs*100:.1f}%)")
                report_lines.append(f"  Medium objects (1-10% of image): {metrics.medium_objects_count} "
                                  f"({metrics.medium_objects_count/total_objs*100:.1f}%)")
                report_lines.append(f"  Large objects (>10% of image): {metrics.large_objects_count} "
                                  f"({metrics.large_objects_count/total_objs*100:.1f}%)")
            
            # Spatial distribution
            if metrics.bbox_centers_x:
                report_lines.append(f"\n[SPATIAL DISTRIBUTION]")
                report_lines.append(f"  Center X - Mean: {np.mean(metrics.bbox_centers_x):.3f}, "
                                  f"Std: {np.std(metrics.bbox_centers_x):.3f}")
                report_lines.append(f"  Center Y - Mean: {np.mean(metrics.bbox_centers_y):.3f}, "
                                  f"Std: {np.std(metrics.bbox_centers_y):.3f}")
                report_lines.append(f"  Objects near edge: {metrics.objects_near_edge} "
                                  f"({metrics.objects_near_edge/total_objs*100:.1f}%)")
            
            # Coverage statistics
            if metrics.image_coverage_ratios:
                report_lines.append(f"\n[COVERAGE STATISTICS]")
                report_lines.append(f"  Image coverage - Mean: {np.mean(metrics.image_coverage_ratios):.3f}, "
                                  f"Median: {np.median(metrics.image_coverage_ratios):.3f}, "
                                  f"Max: {np.max(metrics.image_coverage_ratios):.3f}")
            
            # Overlap statistics
            if metrics.bbox_overlaps:
                report_lines.append(f"\n[OVERLAP STATISTICS]")
                report_lines.append(f"  Number of overlapping pairs: {len(metrics.bbox_overlaps)}")
                report_lines.append(f"  Mean IoU: {np.mean(metrics.bbox_overlaps):.3f}")
                report_lines.append(f"  Max IoU: {np.max(metrics.bbox_overlaps):.3f}")
        
        # Quality assessment
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append("QUALITY ASSESSMENT")
        report_lines.append(f"{'=' * 80}")
        report_lines.extend(self._generate_quality_assessment())
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")
        
        return report
    
    def _generate_quality_assessment(self) -> List[str]:
        """Generate quality assessment and recommendations."""
        lines = []
        
        for split, metrics in self.metrics.items():
            lines.append(f"\n[{split.upper()}]")
            
            # Check for class imbalance
            if metrics.class_counts:
                counts = list(metrics.class_counts.values())
                if len(counts) > 1:
                    imbalance_ratio = max(counts) / min(counts)
                    if imbalance_ratio > 5:
                        lines.append(f"  WARNING: Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                    else:
                        lines.append(f"  OK: Class distribution is balanced (ratio: {imbalance_ratio:.1f}:1)")
            
            # Check for small objects
            total_objs = metrics.small_objects_count + metrics.medium_objects_count + metrics.large_objects_count
            if total_objs > 0:
                small_ratio = metrics.small_objects_count / total_objs
                if small_ratio > 0.3:
                    lines.append(f"  WARNING: High proportion of small objects ({small_ratio*100:.1f}%) - "
                               "may be challenging for detection")
            
            # Check annotations per image
            if metrics.annotations_per_image:
                avg_ann = np.mean(metrics.annotations_per_image)
                if avg_ann < 1:
                    lines.append(f"  WARNING: Low average annotations per image ({avg_ann:.2f})")
                elif avg_ann > 20:
                    lines.append(f"  NOTE: High average annotations per image ({avg_ann:.2f})")
            
            # Check for images without annotations
            if metrics.images_without_annotations > 0:
                ratio = metrics.images_without_annotations / metrics.total_images
                if ratio > 0.1:
                    lines.append(f"  WARNING: {ratio*100:.1f}% of images have no annotations")
            
            # Check spatial bias
            if metrics.bbox_centers_x and metrics.bbox_centers_y:
                center_x_std = np.std(metrics.bbox_centers_x)
                center_y_std = np.std(metrics.bbox_centers_y)
                if center_x_std < 0.2 or center_y_std < 0.2:
                    lines.append(f"  WARNING: Objects may be spatially biased "
                               f"(X std: {center_x_std:.3f}, Y std: {center_y_std:.3f})")
            
            # Check edge proximity
            if total_objs > 0:
                edge_ratio = metrics.objects_near_edge / total_objs
                if edge_ratio > 0.3:
                    lines.append(f"  NOTE: {edge_ratio*100:.1f}% of objects are near image edges")
        
        return lines
    
    def plot_distributions(self, split: str = 'train', output_dir: Optional[str] = None, 
                          figsize: Tuple[int, int] = (20, 12)):
        """
        Generate comprehensive visualization plots for a dataset split.
        
        Args:
            split: Dataset split to visualize
            output_dir: Optional directory to save plots
            figsize: Figure size for the plot grid
        """
        if split not in self.metrics:
            print(f"No metrics available for split: {split}")
            return
        
        metrics = self.metrics[split]
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Class distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics.class_counts:
            classes = list(metrics.class_counts.keys())
            counts = list(metrics.class_counts.values())
            ax1.bar(classes, counts, color='steelblue')
            ax1.set_title('Class Distribution')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Annotations per image
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics.annotations_per_image:
            ax2.hist(metrics.annotations_per_image, bins=30, color='coral', edgecolor='black')
            ax2.set_title('Annotations per Image')
            ax2.set_xlabel('Number of Annotations')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(metrics.annotations_per_image), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(metrics.annotations_per_image):.1f}')
            ax2.legend()
        
        # 3. Bounding box area distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if metrics.bbox_areas:
            ax3.hist(metrics.bbox_areas, bins=50, color='lightgreen', edgecolor='black')
            ax3.set_title('Bounding Box Area Distribution')
            ax3.set_xlabel('Normalized Area')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(metrics.bbox_areas), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(metrics.bbox_areas):.3f}')
            ax3.legend()
        
        # 4. Object size categories
        ax4 = fig.add_subplot(gs[0, 3])
        sizes = ['Small\n(<1%)', 'Medium\n(1-10%)', 'Large\n(>10%)']
        counts_sizes = [metrics.small_objects_count, metrics.medium_objects_count, 
                       metrics.large_objects_count]
        colors_sizes = ['#ff9999', '#ffcc99', '#99cc99']
        ax4.pie(counts_sizes, labels=sizes, autopct='%1.1f%%', colors=colors_sizes, startangle=90)
        ax4.set_title('Object Size Distribution')
        
        # 5. Bounding box aspect ratios
        ax5 = fig.add_subplot(gs[1, 0])
        if metrics.bbox_aspect_ratios:
            ax5.hist(metrics.bbox_aspect_ratios, bins=50, color='plum', edgecolor='black')
            ax5.set_title('Bounding Box Aspect Ratios')
            ax5.set_xlabel('Aspect Ratio (Width/Height)')
            ax5.set_ylabel('Frequency')
            ax5.axvline(np.median(metrics.bbox_aspect_ratios), color='red', 
                       linestyle='--', label=f'Median: {np.median(metrics.bbox_aspect_ratios):.2f}')
            ax5.legend()
        
        # 6. Rotation angles
        ax6 = fig.add_subplot(gs[1, 1])
        if metrics.bbox_angles:
            ax6.hist(metrics.bbox_angles, bins=36, color='skyblue', edgecolor='black')
            ax6.set_title('Bounding Box Rotation Angles')
            ax6.set_xlabel('Angle (degrees)')
            ax6.set_ylabel('Frequency')
        
        # 7. Spatial distribution heatmap
        ax7 = fig.add_subplot(gs[1, 2])
        if metrics.bbox_centers_x and metrics.bbox_centers_y:
            heatmap, xedges, yedges = np.histogram2d(
                metrics.bbox_centers_x, metrics.bbox_centers_y, bins=20
            )
            extent = [0, 1, 0, 1]
            im = ax7.imshow(heatmap.T, origin='lower', extent=extent, cmap='YlOrRd', aspect='auto')
            ax7.set_title('Spatial Distribution of Object Centers')
            ax7.set_xlabel('Normalized X')
            ax7.set_ylabel('Normalized Y')
            plt.colorbar(im, ax=ax7, label='Density')
        
        # 8. Image coverage ratios
        ax8 = fig.add_subplot(gs[1, 3])
        if metrics.image_coverage_ratios:
            ax8.hist(metrics.image_coverage_ratios, bins=30, color='gold', edgecolor='black')
            ax8.set_title('Image Coverage Ratios')
            ax8.set_xlabel('Coverage Ratio')
            ax8.set_ylabel('Frequency')
            ax8.axvline(np.mean(metrics.image_coverage_ratios), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(metrics.image_coverage_ratios):.3f}')
            ax8.legend()
        
        # 9. Width vs Height scatter
        ax9 = fig.add_subplot(gs[2, 0])
        if metrics.bbox_widths and metrics.bbox_heights:
            ax9.scatter(metrics.bbox_widths, metrics.bbox_heights, alpha=0.5, s=10, color='purple')
            ax9.set_title('BBox Width vs Height')
            ax9.set_xlabel('Width (normalized)')
            ax9.set_ylabel('Height (normalized)')
            ax9.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Square aspect')
            ax9.legend()
        
        # 10. Image aspect ratios
        ax10 = fig.add_subplot(gs[2, 1])
        if metrics.image_aspect_ratios:
            ax10.hist(metrics.image_aspect_ratios, bins=30, color='teal', edgecolor='black')
            ax10.set_title('Image Aspect Ratios')
            ax10.set_xlabel('Aspect Ratio')
            ax10.set_ylabel('Frequency')
        
        # 11. IoU overlap distribution
        ax11 = fig.add_subplot(gs[2, 2])
        if metrics.bbox_overlaps:
            ax11.hist(metrics.bbox_overlaps, bins=30, color='salmon', edgecolor='black')
            ax11.set_title('Bounding Box Overlaps (IoU)')
            ax11.set_xlabel('IoU')
            ax11.set_ylabel('Frequency')
            ax11.axvline(np.mean(metrics.bbox_overlaps), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(metrics.bbox_overlaps):.3f}')
            ax11.legend()
        else:
            ax11.text(0.5, 0.5, 'No overlaps detected', 
                     ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('Bounding Box Overlaps (IoU)')
        
        # 12. Image file sizes
        ax12 = fig.add_subplot(gs[2, 3])
        if metrics.image_sizes_mb:
            ax12.hist(metrics.image_sizes_mb, bins=30, color='lightblue', edgecolor='black')
            ax12.set_title('Image File Sizes')
            ax12.set_xlabel('Size (MB)')
            ax12.set_ylabel('Frequency')
        
        # 13. Box plot of bbox dimensions
        ax13 = fig.add_subplot(gs[3, 0])
        if metrics.bbox_widths and metrics.bbox_heights:
            data_to_plot = [metrics.bbox_widths, metrics.bbox_heights]
            ax13.boxplot(data_to_plot, labels=['Width', 'Height'])
            ax13.set_title('BBox Dimensions Distribution')
            ax13.set_ylabel('Normalized Size')
        
        # 14. Cumulative distribution of areas
        ax14 = fig.add_subplot(gs[3, 1])
        if metrics.bbox_areas:
            sorted_areas = np.sort(metrics.bbox_areas)
            cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
            ax14.plot(sorted_areas, cumulative, color='darkgreen', linewidth=2)
            ax14.set_title('Cumulative Distribution of Areas')
            ax14.set_xlabel('Normalized Area')
            ax14.set_ylabel('Cumulative Probability')
            ax14.grid(True, alpha=0.3)
        
        # 15. Annotations statistics summary
        ax15 = fig.add_subplot(gs[3, 2:])
        ax15.axis('off')
        
        summary_text = f"""
        DATASET SUMMARY - {split.upper()} SPLIT
        
        Total Images: {metrics.total_images}
        Total Annotations: {metrics.total_annotations}
        Images with Annotations: {metrics.images_with_annotations}
        
        Annotations per Image: {np.mean(metrics.annotations_per_image):.2f} ± {np.std(metrics.annotations_per_image):.2f}
        
        Bounding Box Metrics:
          - Mean Area: {np.mean(metrics.bbox_areas):.4f}
          - Mean Width: {np.mean(metrics.bbox_widths):.4f}
          - Mean Height: {np.mean(metrics.bbox_heights):.4f}
          - Mean Aspect Ratio: {np.mean(metrics.bbox_aspect_ratios):.2f}
        
        Object Sizes:
          - Small (<1%): {metrics.small_objects_count}
          - Medium (1-10%): {metrics.medium_objects_count}
          - Large (>10%): {metrics.large_objects_count}
        """
        
        ax15.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        plt.suptitle(f'Dataset Analysis - {split.upper()} Split', fontsize=16, fontweight='bold')
        
        if output_dir:
            output_path = Path(output_dir) / f'{split}_analysis.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        plt.show()
    
    def plot_sample_images(self, split: str = 'train', num_samples: int = 6,
                          output_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Display sample images with annotations overlaid.
        
        Args:
            split: Dataset split to sample from
            num_samples: Number of sample images to display
            output_path: Optional path to save the figure
            figsize: Figure size
        """
        # Determine paths
        if split == 'train':
            img_dir = self.dataset_path / self.config.get('train', f'images/{split}')
            label_dir = self.dataset_path / 'labels' / split
        elif split == 'val':
            img_dir = self.dataset_path / self.config.get('val', f'images/{split}')
            label_dir = self.dataset_path / 'labels' / split
        elif split == 'test':
            img_dir = self.dataset_path / self.config.get('test', f'images/{split}')
            label_dir = self.dataset_path / 'labels' / split
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Get sample images with annotations
        image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        samples = []
        
        for img_path in image_files:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                annotations = self._parse_label_file(label_path)
                if annotations:
                    samples.append((img_path, label_path, annotations))
            
            if len(samples) >= num_samples:
                break
        
        if not samples:
            print(f"No annotated images found in {split} split")
            return
        
        # Create figure
        rows = (num_samples + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        class_names = self.config.get('names', {})
        
        for idx, (img_path, label_path, annotations) in enumerate(samples):
            if idx >= len(axes):
                break
            
            # Load and display image
            img = Image.open(img_path)
            axes[idx].imshow(img)
            
            # Overlay bounding boxes
            width, height = img.size
            
            for ann in annotations:
                points = np.array(ann['points']).reshape(4, 2)
                
                # Convert normalized to pixel coordinates
                points[:, 0] *= width
                points[:, 1] *= height
                
                # Draw polygon
                from matplotlib.patches import Polygon
                poly = Polygon(points, fill=False, edgecolor='red', linewidth=2)
                axes[idx].add_patch(poly)
                
                # Add class label
                class_name = class_names.get(ann['class_id'], f"class_{ann['class_id']}")
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])
                axes[idx].text(center_x, center_y, class_name, 
                             color='white', fontsize=8, fontweight='bold',
                             bbox=dict(facecolor='red', alpha=0.7))
            
            axes[idx].set_title(f"{img_path.name}\n{len(annotations)} annotations", fontsize=8)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Sample Images - {split.upper()} Split', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Sample images saved to: {output_path}")
        
        plt.show()
    
    def export_metrics(self, output_path: str, format: str = 'json'):
        """
        Export computed metrics to a file.
        
        Args:
            output_path: Path to save the metrics
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            data = {split: metrics.to_dict() for split, metrics in self.metrics.items()}
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Metrics exported to: {output_path}")
        
        elif format == 'csv':
            # Flatten metrics for CSV export
            rows = []
            for split, metrics in self.metrics.items():
                row = {'split': split}
                row.update(metrics.to_dict())
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"Metrics exported to: {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_metrics_dataframe(self, split: str = 'train') -> pd.DataFrame:
        """
        Get metrics as a pandas DataFrame for interactive analysis.
        
        Args:
            split: Dataset split
            
        Returns:
            DataFrame with per-annotation metrics
        """
        if split not in self.metrics:
            raise ValueError(f"No metrics for split: {split}")
        
        metrics = self.metrics[split]
        
        data = {
            'bbox_area': metrics.bbox_areas,
            'bbox_width': metrics.bbox_widths,
            'bbox_height': metrics.bbox_heights,
            'bbox_aspect_ratio': metrics.bbox_aspect_ratios,
            'bbox_angle': metrics.bbox_angles,
            'center_x': metrics.bbox_centers_x,
            'center_y': metrics.bbox_centers_y,
        }
        
        df = pd.DataFrame(data)
        return df


def main():
    """Main function for standalone script usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze YOLO OBB dataset quality and generate comprehensive reports'
    )
    parser.add_argument('dataset_path', type=str, help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, default='dataset_analysis',
                       help='Output directory for reports and visualizations')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip loading image metadata (faster but less detailed)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Splits to analyze')
    parser.add_argument('--export-format', choices=['json', 'csv'], default='json',
                       help='Format for exporting metrics')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    print(f"Initializing analyzer for: {args.dataset_path}")
    analyzer = YOLOOBBDatasetAnalyzer(
        dataset_path=args.dataset_path,
        load_images=not args.no_images
    )
    
    # Analyze all splits
    print("\nAnalyzing dataset splits...")
    analyzer.analyze_all_splits()
    
    # Generate report
    print("\nGenerating text report...")
    report = analyzer.generate_report(output_path=output_dir / 'dataset_report.txt')
    print(report)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    for split in analyzer.metrics.keys():
        analyzer.plot_distributions(
            split=split,
            output_dir=output_dir
        )
    
    # Export metrics
    print("\nExporting metrics...")
    analyzer.export_metrics(
        output_path=output_dir / f'metrics.{args.export_format}',
        format=args.export_format
    )
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

