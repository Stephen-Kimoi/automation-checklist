#!/usr/bin/env python3
"""
Configuration file for batch processing parameters
"""

# Batch processing configuration
BATCH_CONFIG = {
    # Default batch percentage (10% of total projects per batch)
    'default_batch_percentage': 10.0,
    
    # Recommended batch percentages for different dataset sizes
    'recommended_batch_sizes': {
        'small': {  # 50-200 projects
            'batch_percentage': 25.0,
            'description': '25% batches for small datasets'
        },
        'medium': {  # 200-500 projects
            'batch_percentage': 15.0,
            'description': '15% batches for medium datasets'
        },
        'large': {  # 500-1000 projects
            'batch_percentage': 10.0,
            'description': '10% batches for large datasets'
        },
        'very_large': {  # 1000+ projects
            'batch_percentage': 5.0,
            'description': '5% batches for very large datasets'
        }
    },
    
    # Output directory structure
    'output_structure': {
        'batch_prefix': 'batch_',
        'combined_prefix': 'combined_results_',
        'final_filename': 'final_results.csv',
        'report_suffix': '_detailed_report.txt'
    },
    
    # Progress reporting
    'progress_reporting': {
        'show_batch_progress': True,
        'show_individual_progress': True,
        'progress_bar_width': 60
    }
}

def get_recommended_batch_size(total_projects: int) -> float:
    """Get recommended batch percentage based on total project count"""
    if total_projects < 200:
        return BATCH_CONFIG['recommended_batch_sizes']['small']['batch_percentage']
    elif total_projects < 500:
        return BATCH_CONFIG['recommended_batch_sizes']['medium']['batch_percentage']
    elif total_projects < 1000:
        return BATCH_CONFIG['recommended_batch_sizes']['large']['batch_percentage']
    else:
        return BATCH_CONFIG['recommended_batch_sizes']['very_large']['batch_percentage']

def get_batch_info(total_projects: int, batch_percentage: float = None) -> dict:
    """Get batch processing information"""
    if batch_percentage is None:
        batch_percentage = get_recommended_batch_size(total_projects)
    
    batch_size = max(1, int(total_projects * (batch_percentage / 100.0)))
    num_batches = (total_projects + batch_size - 1) // batch_size
    
    return {
        'total_projects': total_projects,
        'batch_percentage': batch_percentage,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'estimated_time_per_batch': '5-15 minutes',  # Rough estimate
        'total_estimated_time': f"{num_batches * 10}-{num_batches * 15} minutes"
    }

if __name__ == "__main__":
    # Example usage
    print("Batch Processing Configuration")
    print("=" * 40)
    
    test_sizes = [100, 300, 600, 1200]
    
    for size in test_sizes:
        info = get_batch_info(size)
        print(f"\nFor {size} projects:")
        print(f"  Recommended batch size: {info['batch_percentage']}%")
        print(f"  Projects per batch: {info['batch_size']}")
        print(f"  Number of batches: {info['num_batches']}")
        print(f"  Estimated total time: {info['total_estimated_time']}")
