#!/usr/bin/env python3
"""
ICP Project Evaluation System - Main Script

This script evaluates ICP projects based on README quality and commit activity.
"""

import argparse
import sys
from evaluator import ICPProjectEvaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate ICP projects from CSV file')
    parser.add_argument('input_csv', help='Path to input CSV file containing repo URLs (or filename in input_csv folder)')
    parser.add_argument('output_csv', help='Path to output CSV file for results (or filename in output_csv folder)')
    parser.add_argument('--hackathon-start', default='2024-07-01', 
                       help='Hackathon start date (YYYY-MM-DD)')
    parser.add_argument('--hackathon-end', default='2024-12-31',
                       help='Hackathon end date (YYYY-MM-DD)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating detailed report (CSV only)')
    
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
        
        print("Updating hackathon dates if provided:")
        if args.hackathon_start != '2024-07-01':
            from datetime import datetime
            evaluator.hackathon_start = datetime.strptime(args.hackathon_start, '%Y-%m-%d').replace(tzinfo=None)
        if args.hackathon_end != '2024-12-31':
            from datetime import datetime
            evaluator.hackathon_end = datetime.strptime(args.hackathon_end, '%Y-%m-%d').replace(tzinfo=None)
        
        print(f"Hackathon period: {evaluator.hackathon_start.strftime('%Y-%m-%d')} to {evaluator.hackathon_end.strftime('%Y-%m-%d')}")
        
        print(f"Starting evaluation of projects from: {input_path}")
        
        results = evaluator.evaluate_projects_from_csv(input_path, output_path, generate_report=not args.no_report)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total projects evaluated: {len(results)}")
        print(f"Average total score: {results['total_score'].mean():.2f}")
        print(f"Average README installation score: {results['readme_installation_score'].mean():.2f}")
        print(f"Average README quality score: {results['readme_quality_score'].mean():.2f}")
        print(f"Average commit activity score: {results['commit_activity_score'].mean():.2f}")
        
        print("\nTop 3 Projects by Total Score:")
        
        top_projects = results.nlargest(3, 'total_score')
        for idx, (_, project) in enumerate(top_projects.iterrows(), 1):
            print(f"{idx}. {project['project_name']} - Score: {project['total_score']}")
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 