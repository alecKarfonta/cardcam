#!/usr/bin/env python3
"""
Demo script showing how to use the Dataset Analyzer in various ways.

This script demonstrates the main features of the YOLOOBBDatasetAnalyzer class
without requiring actual execution (good for documentation and examples).
"""

# Note: This is a demo/documentation script. It shows example code
# but doesn't actually execute it (to avoid dependency requirements)


def demo_basic_analysis():
    """Basic analysis workflow."""
    print("=" * 80)
    print("DEMO 1: Basic Analysis")
    print("=" * 80)
    
    print("""
    # Initialize analyzer
    analyzer = YOLOOBBDatasetAnalyzer('/path/to/dataset', load_images=True)
    
    # Analyze all splits
    results = analyzer.analyze_all_splits()
    
    # Generate report
    report = analyzer.generate_report(output_path='report.txt')
    print(report)
    """)


def demo_visualizations():
    """Visualization examples."""
    print("\n" + "=" * 80)
    print("DEMO 2: Generating Visualizations")
    print("=" * 80)
    
    print("""
    analyzer = YOLOOBBDatasetAnalyzer('/path/to/dataset')
    analyzer.analyze_all_splits()
    
    # Generate comprehensive visualization for training set
    analyzer.plot_distributions(
        split='train',
        output_dir='./visualizations',
        figsize=(20, 12)
    )
    
    # Display sample images with annotations
    analyzer.plot_sample_images(
        split='train',
        num_samples=9,
        output_path='samples.png'
    )
    """)


def demo_interactive_analysis():
    """Interactive analysis with pandas."""
    print("\n" + "=" * 80)
    print("DEMO 3: Interactive Analysis with Pandas")
    print("=" * 80)
    
    print("""
    analyzer = YOLOOBBDatasetAnalyzer('/path/to/dataset')
    analyzer.analyze_all_splits()
    
    # Get metrics as DataFrame
    df = analyzer.get_metrics_dataframe(split='train')
    
    # Display basic info
    print(df.describe())
    
    # Find small objects with extreme aspect ratios
    small_extreme = df[
        (df['bbox_area'] < 0.01) & 
        ((df['bbox_aspect_ratio'] < 0.5) | (df['bbox_aspect_ratio'] > 2.0))
    ]
    print(f"Found {len(small_extreme)} small objects with extreme aspect ratios")
    
    # Compute correlations
    correlations = df.corr()
    print(correlations)
    """)


def demo_export_metrics():
    """Export metrics examples."""
    print("\n" + "=" * 80)
    print("DEMO 4: Exporting Metrics")
    print("=" * 80)
    
    print("""
    analyzer = YOLOOBBDatasetAnalyzer('/path/to/dataset')
    analyzer.analyze_all_splits()
    
    # Export as JSON
    analyzer.export_metrics('metrics.json', format='json')
    
    # Export as CSV
    analyzer.export_metrics('metrics.csv', format='csv')
    
    # Save report
    analyzer.generate_report(output_path='full_report.txt')
    """)


def demo_custom_analysis():
    """Custom analysis examples."""
    print("\n" + "=" * 80)
    print("DEMO 5: Custom Analysis")
    print("=" * 80)
    
    print("""
    analyzer = YOLOOBBDatasetAnalyzer('/path/to/dataset')
    analyzer.analyze_all_splits()
    
    # Access raw metrics
    train_metrics = analyzer.metrics['train']
    
    # Custom computation: find outliers
    import numpy as np
    areas = np.array(train_metrics.bbox_areas)
    q1, q3 = np.percentile(areas, [25, 75])
    iqr = q3 - q1
    outliers = areas[(areas < q1 - 1.5*iqr) | (areas > q3 + 1.5*iqr)]
    print(f"Found {len(outliers)} area outliers")
    
    # Compare splits
    train_avg_area = np.mean(train_metrics.bbox_areas)
    val_metrics = analyzer.metrics['val']
    val_avg_area = np.mean(val_metrics.bbox_areas)
    print(f"Train avg area: {train_avg_area:.4f}")
    print(f"Val avg area: {val_avg_area:.4f}")
    """)


def demo_comparison():
    """Compare multiple datasets."""
    print("\n" + "=" * 80)
    print("DEMO 6: Comparing Multiple Datasets")
    print("=" * 80)
    
    print("""
    # Analyze multiple datasets
    datasets = {
        'gold': '/path/to/gold',
        'synthetic': '/path/to/synthetic',
        'merged': '/path/to/merged'
    }
    
    results = {}
    for name, path in datasets.items():
        analyzer = YOLOOBBDatasetAnalyzer(path)
        analyzer.analyze_all_splits()
        results[name] = analyzer.metrics
    
    # Compare key metrics
    import pandas as pd
    comparison = []
    for ds_name, metrics in results.items():
        if 'train' in metrics:
            m = metrics['train']
            comparison.append({
                'Dataset': ds_name,
                'Images': m.total_images,
                'Annotations': m.total_annotations,
                'Avg Annotations/Image': m.total_annotations / m.total_images,
                'Avg BBox Area': sum(m.bbox_areas) / len(m.bbox_areas) if m.bbox_areas else 0,
                'Small Objects %': m.small_objects_count / 
                    (m.small_objects_count + m.medium_objects_count + m.large_objects_count) * 100
            })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df)
    """)


def demo_quality_checks():
    """Quality assessment examples."""
    print("\n" + "=" * 80)
    print("DEMO 7: Quality Assessment")
    print("=" * 80)
    
    print("""
    analyzer = YOLOOBBDatasetAnalyzer('/path/to/dataset')
    analyzer.analyze_all_splits()
    
    # The analyzer automatically performs quality checks
    # Check the report for warnings:
    report = analyzer.generate_report()
    
    # Warnings include:
    # - Class imbalance (>5:1 ratio)
    # - High proportion of small objects (>30%)
    # - Low annotations per image (<1)
    # - Many images without annotations (>10%)
    # - Spatial bias (low std dev in centers)
    # - Many objects near edges (>30%)
    
    # Access quality metrics directly
    train_metrics = analyzer.metrics['train']
    total_objects = (train_metrics.small_objects_count + 
                    train_metrics.medium_objects_count + 
                    train_metrics.large_objects_count)
    
    if total_objects > 0:
        small_ratio = train_metrics.small_objects_count / total_objects
        if small_ratio > 0.3:
            print(f"WARNING: {small_ratio*100:.1f}% of objects are small")
        else:
            print(f"OK: Only {small_ratio*100:.1f}% of objects are small")
    """)


def demo_fast_analysis():
    """Fast analysis for large datasets."""
    print("\n" + "=" * 80)
    print("DEMO 8: Fast Analysis (Large Datasets)")
    print("=" * 80)
    
    print("""
    # For very large datasets, skip image loading
    analyzer = YOLOOBBDatasetAnalyzer(
        '/path/to/large/dataset',
        load_images=False  # Skip loading image metadata
    )
    
    # This will be much faster but won't include:
    # - Image dimensions
    # - Image aspect ratios  
    # - File sizes
    
    analyzer.analyze_all_splits()
    
    # All other metrics (based on labels) will still be computed
    report = analyzer.generate_report()
    """)


def demo_extended_class():
    """Extending the analyzer class."""
    print("\n" + "=" * 80)
    print("DEMO 9: Extending the Analyzer")
    print("=" * 80)
    
    print("""
    from dataset_analyzer import YOLOOBBDatasetAnalyzer, DatasetMetrics
    
    class CustomAnalyzer(YOLOOBBDatasetAnalyzer):
        '''Custom analyzer with additional features.'''
        
        def compute_custom_metric(self, split='train'):
            '''Add your custom metric computation.'''
            metrics = self.metrics[split]
            
            # Example: compute average object density per image
            if metrics.image_coverage_ratios:
                density = sum(metrics.image_coverage_ratios) / len(metrics.image_coverage_ratios)
                return density
            return 0.0
        
        def plot_custom_visualization(self, split='train'):
            '''Add your custom plots.'''
            import matplotlib.pyplot as plt
            metrics = self.metrics[split]
            
            # Your custom plotting code here
            plt.figure(figsize=(10, 6))
            # ... plotting logic ...
            plt.show()
    
    # Use the custom analyzer
    analyzer = CustomAnalyzer('/path/to/dataset')
    analyzer.analyze_all_splits()
    density = analyzer.compute_custom_metric(split='train')
    print(f"Average object density: {density:.3f}")
    """)


def demo_notebook_integration():
    """Using in Jupyter notebooks."""
    print("\n" + "=" * 80)
    print("DEMO 10: Jupyter Notebook Integration")
    print("=" * 80)
    
    print("""
    # In a Jupyter notebook cell:
    
    from dataset_analyzer import YOLOOBBDatasetAnalyzer
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    # Initialize
    analyzer = YOLOOBBDatasetAnalyzer('../data/dataset')
    analyzer.analyze_all_splits()
    
    # Get interactive DataFrame
    df = analyzer.get_metrics_dataframe('train')
    
    # Use pandas plotting
    df['bbox_area'].hist(bins=50)
    plt.title('Area Distribution')
    plt.show()
    
    # Use seaborn for advanced plots
    import seaborn as sns
    sns.jointplot(data=df, x='bbox_width', y='bbox_height', kind='hex')
    plt.show()
    
    # Interactive filtering
    small_objects = df[df['bbox_area'] < 0.01]
    print(f"Small objects: {len(small_objects)}")
    
    # Statistical tests
    from scipy.stats import ks_2samp
    df_val = analyzer.get_metrics_dataframe('val')
    stat, pval = ks_2samp(df['bbox_area'], df_val['bbox_area'])
    print(f"Train vs Val area distribution similarity: p={pval:.4f}")
    """)


def main():
    """Run all demos."""
    print("\n")
    print("#" * 80)
    print("# DATASET ANALYZER - DEMO EXAMPLES")
    print("#" * 80)
    print("\nThis script shows various ways to use the YOLOOBBDatasetAnalyzer class.")
    print("Copy and modify these examples for your specific use case.\n")
    
    demo_basic_analysis()
    demo_visualizations()
    demo_interactive_analysis()
    demo_export_metrics()
    demo_custom_analysis()
    demo_comparison()
    demo_quality_checks()
    demo_fast_analysis()
    demo_extended_class()
    demo_notebook_integration()
    
    print("\n" + "#" * 80)
    print("# END OF DEMOS")
    print("#" * 80)
    print("\nFor actual usage, see:")
    print("  - README.md for full documentation")
    print("  - QUICKSTART.md for quick start guide")
    print("  - dataset_analysis_example.ipynb for interactive examples")
    print()


if __name__ == '__main__':
    main()

