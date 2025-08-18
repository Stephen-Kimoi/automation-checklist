#!/usr/bin/env python3
"""
Test script for the ICP Project Evaluator
"""

import os
from evaluator import ICPProjectEvaluator

def test_single_project():
    """Test evaluation of a single project."""
    
    # Check if API keys are set
    if not os.getenv('GROQ_API_KEY') or not os.getenv('GITHUB_TOKEN'):
        print("Please set GROQ_API_KEY and GITHUB_TOKEN environment variables")
        print("You can copy config.env.example to .env and add your keys")
        return
    
    # Initialize evaluator
    evaluator = ICPProjectEvaluator()
    
    # Test with a well-known repository
    test_repo = "https://github.com/dfinity/portal"
    
    print(f"Testing evaluation with: {test_repo}")
    print("="*50)
    
    try:
        result = evaluator.evaluate_project(test_repo)
        
        print("EVALUATION RESULTS:")
        print(f"Project: {result['project_name']}")
        print(f"GitHub Link: {result['github_link']}")
        print(f"README Documentation Score: {result['readme_documentation_score']}/5")
        print(f"Commit Activity Score: {result['commit_activity_score']}/3")
        print(f"Total Score: {result['total_score']}/8")
        print(f"README Documentation Comments: {result['readme_documentation_comments']}")
        print(f"Commit Activity Comments: {result['commit_activity_comments']}")
        
        print("\n" + "="*50)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_project() 