# Automated Evaluation System

## Overview

This system evaluates ICP projects based on two main criteria:

1. **README Documentation Quality (5 points)**: Evaluates whether the README includes setup instructions (for local dev), general project description, integration guide (if applicable), and contribution guidelines.

2. **Commit Activity (3 points)**: Analyzes weekly commit patterns during the hackathon period:
   - 0 points: No commits
   - 1 point: 1 or 2 commits total
   - 2 points: Commits every other week
   - 3 points: Commits every week (with 2+ commits per week)

The system also generates weekly summaries of what features were built or improved based on commit messages.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp config.env.example .env
```

Edit `.env` file and add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
GITHUB_TOKEN=your_github_token_here
```

## Usage

### Basic Usage

1. Create a virtual environment: 
```bash
python3 -m venv venv 
```

2. Activate the virtual environment: 
```bash
source venv/bin/activate 
```

3. Install requirements: 
```bash
pip3 install -r requirements.txt   
```

3. Run the script: 
```bash
python main.py input.csv output.csv
```

### Custom Hackathon Dates

```bash
python main.py input.csv output.csv --hackathon-start 2024-07-01 --hackathon-end 2024-12-31
```

## Running files: 

Run test script with 
```bash
python test_evaluator.py
``` 

Running full evaluation script with 
```bash
python main.py sample_input.csv results.csv
``` 

## Input CSV Format

The input CSV should contain a `repo_url` column with GitHub repository URLs to evaluate.

## Output

The system generates:
- A CSV file with evaluation results
- A detailed text report with project-by-project analysis and weekly development summaries

