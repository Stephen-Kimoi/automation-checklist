# Automated Evaluation System

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

