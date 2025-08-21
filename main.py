#!/usr/bin/env python3
"""
ICP Project Evaluation System - Main Script

This script evaluates ICP projects based on README documentation quality and commit activity.
"""

import argparse
import sys
from evaluator import ICPProjectEvaluator
from datetime import datetime, timezone

def main():
    parser = argparse.ArgumentParser(description='Evaluate ICP projects from CSV file')
    parser.add_argument('input_csv', help='Path to input CSV file containing repo URLs (or filename in input_csv folder)')
    parser.add_argument('output_csv', help='Path to output CSV file for results (or filename in output_csv folder)')
    parser.add_argument('--hackathon-start', default='2025-08-01', 
                       help='Hackathon start date (YYYY-MM-DD)')
    parser.add_argument('--hackathon-end', default='2025-08-23',
                       help='Hackathon end date (YYYY-MM-DD)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating detailed report (CSV only)')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Enable batch processing mode for large datasets')
    parser.add_argument('--batch-percentage', type=float, default=10.0,
                       help='Percentage of total projects to process per batch (default: 10%%)')
    parser.add_argument('--resume-from', type=str,
                       help='Path to resume from a previous batch (for batch mode)')
    
    args = parser.parse_args()
    
    try:
        input_path = args.input_csv
        output_path = args.output_csv
        
        if not input_path.startswith('/') and not input_path.startswith('./') and not input_path.startswith('../'):
            input_path = f"input_csv/{input_path}"
        
        if not output_path.startswith('/') and not output_path.startswith('./') and not output_path.startswith('../'):
            output_path = f"output_csv/{output_path}"
        
        print("Initializing ICP Project Evaluator...")
        evaluator = ICPProjectEvaluator()
        
        print("Always set hackathon period as UTC-aware...")
        evaluator.hackathon_start = datetime.strptime(args.hackathon_start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        print("Hackathon start date: ", evaluator.hackathon_start)
        evaluator.hackathon_end = datetime.strptime(args.hackathon_end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        print("Hackathon end date: ", evaluator.hackathon_end)
        
        print(f"Hackathon period: {evaluator.hackathon_start.strftime('%Y-%m-%d')} to {evaluator.hackathon_end.strftime('%Y-%m-%d')}")
        
        print(f"Starting evaluation of projects from: {input_path}")
        
        if args.batch_mode:
            # Use batch processing mode
            print(f"BATCH MODE ENABLED - Processing in batches of {args.batch_percentage}%% of total projects")
            
            # For batch mode, output_path should be a directory
            if not output_path.endswith('/'):
                output_path = output_path + '/'
            
            results_path = evaluator.evaluate_projects_in_batches(
                input_path, 
                output_path, 
                batch_percentage=args.batch_percentage,
                generate_report=not args.no_report,
                resume_from=args.resume_from
            )
            
            print(f"\nBatch processing completed! Final results saved to: {results_path}")
            
        else:
            # Use original single-batch mode
            print("Using standard evaluation mode (all projects at once)")
            results = evaluator.evaluate_projects_from_csv(input_path, output_path, generate_report=not args.no_report)
            
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            print("SCORING BREAKDOWN:")
            print("  readme_documentation_score (out of 5)")
            print("  commit_activity_score (out of 3)")
            print("  total_score (out of 8)")
            print("="*50)
            print(f"Total projects evaluated: {len(results)}")
            print(f"Average scores:")
            print(f"  readme_documentation_score: {results['readme_documentation_score'].mean():.2f}")
            print(f"  commit_activity_score: {results['commit_activity_score'].mean():.2f}")
            print(f"  total_score: {results['total_score'].mean():.2f}")
            
            print("\nTop 3 Projects by Total Score:")
            
            top_projects = results.nlargest(3, 'total_score')
            for idx, (_, project) in enumerate(top_projects.iterrows(), 1):
                print(f"{idx}. {project['project_name']} - Score: {project['total_score']}/8")
            
            print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 