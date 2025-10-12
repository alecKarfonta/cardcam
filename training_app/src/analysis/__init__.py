"""
Dataset Analysis Module

Comprehensive tools for analyzing YOLO OBB format datasets,
computing quality metrics, and generating visualizations.
"""

from .dataset_analyzer import YOLOOBBDatasetAnalyzer, DatasetMetrics

__version__ = "1.0.0"
__all__ = ["YOLOOBBDatasetAnalyzer", "DatasetMetrics"]

