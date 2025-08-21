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

### Batch Processing for Large Datasets

For large datasets (500+ projects), use batch processing mode to evaluate projects in smaller chunks and get incremental results:

```bash
# Process in batches of 10% of total projects (default)
python main.py large_dataset.csv output_folder/ --batch-mode

# Process in batches of 20% of total projects
python main.py large_dataset.csv output_folder/ --batch-mode --batch-percentage 20.0

# Resume from a previous batch (useful if processing was interrupted)
python main.py large_dataset.csv output_folder/ --batch-mode --batch-percentage 10.0 --resume-from output_folder/combined_results_150_projects.csv
```

**Batch Processing Benefits:**
- **Incremental Results**: Get results after each batch instead of waiting for all projects
- **Progress Tracking**: Monitor progress with detailed batch-by-batch updates
- **Resume Capability**: Continue from where you left off if processing is interrupted
- **Memory Efficient**: Process manageable chunks instead of loading all results into memory
- **Percentage-Based**: Automatically calculates batch size based on total project count

**Batch Output Structure:**
```
output_folder/
├── batch_001_results.csv          # Results from first batch
├── batch_001_results_detailed_report.txt
├── combined_results_50_projects.csv    # Combined results after first batch
├── combined_results_50_projects_detailed_report.txt
├── batch_002_results.csv          # Results from second batch
├── batch_002_results_detailed_report.txt
├── combined_results_100_projects.csv   # Combined results after second batch
├── combined_results_100_projects_detailed_report.txt
└── final_results.csv              # Final combined results
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

Test the new batching functionality:
```bash
python test_batching.py
```

Test the rate limiting detection:
```bash
python test_rate_limit_detection.py
```

## Input CSV Format

The input CSV should contain a `repo_url` column with GitHub repository URLs to evaluate.

## Output

The system generates:
- A CSV file with evaluation results
- A detailed text report with project-by-project analysis and weekly development summaries

**In batch mode, it also generates:**
- Individual batch result files
- Incremental combined result files
- Batch-specific detailed reports
- Progress tracking and resume capability
- **Rate limiting detection and warnings**
- **Automatic progress saving when rate limits are hit**

## Rate Limiting Management

The system automatically detects GitHub API rate limiting and provides:

- **Real-time rate limit monitoring** - Shows remaining API requests at startup and during processing
- **Early warnings** - Alerts when API requests are running low
- **Automatic stopping** - Pauses processing when rate limits are hit to prevent wasted time
- **Progress preservation** - Saves all completed work before stopping
- **Resume instructions** - Provides exact commands to resume from where you left off

**When rate limits are hit:**
1. Processing stops immediately
2. Current progress is saved
3. Clear instructions for resuming are displayed
4. You can wait for rate limits to reset or use a different token

**Resume after rate limit reset:**
```bash
python main.py your_dataset.csv output_folder/ --batch-mode --resume-from output_folder/combined_results_50_projects.csv
```

