import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from github import Github
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class ICPProjectEvaluator:
    def __init__(self):
        """Initialize the evaluator with API keys and models."""
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        
        # Initialize GitHub API
        self.github = Github(self.github_token)
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama3-8b-8192"
        )
        
        # Hackathon period (adjust as needed)
        self.hackathon_start = datetime(2024, 7, 1).replace(tzinfo=None)
        self.hackathon_end = datetime.now().replace(tzinfo=None)  # Use current date as hackathon end
    
    def extract_repo_info(self, repo_url: str) -> Tuple[str, str]:
        """Extract owner and repo name from GitHub URL."""
        # Handle different GitHub URL formats
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        
        # Remove .git extension if present
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        parts = repo_url.split('/')
        if 'github.com' in parts:
            github_index = parts.index('github.com')
            owner = parts[github_index + 1]
            repo_name = parts[github_index + 2]
        else:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        return owner, repo_name
    
    def get_readme_content(self, owner: str, repo_name: str) -> str:
        """Fetch README content from GitHub repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            readme = repo.get_readme()
            content = readme.decoded_content.decode('utf-8')
            
            # Truncate content if it's too large (to avoid Groq API limits)
            # Keep first 4000 characters which should be enough for evaluation
            if len(content) > 4000:
                content = content[:4000] + "\n\n[Content truncated due to size limits]"
            
            return content
        except Exception as e:
            print(f"Error fetching README for {owner}/{repo_name}: {e}")
            return ""
    
    def get_commit_history(self, owner: str, repo_name: str) -> List[Dict]:
        """Fetch commit history during hackathon period."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            commits = repo.get_commits(since=self.hackathon_start, until=self.hackathon_end)
            
            commit_data = []
            for commit in commits:
                # Convert timezone-aware datetime to naive datetime for comparison
                commit_date = commit.commit.author.date.replace(tzinfo=None)
                commit_data.append({
                    'sha': commit.sha,
                    'date': commit_date,
                    'message': commit.commit.message,
                    'author': commit.commit.author.name
                })
            
            return commit_data
        except Exception as e:
            print(f"Error fetching commits for {owner}/{repo_name}: {e}")
            return []
    
    def evaluate_readme_installation(self, readme_content: str) -> Tuple[int, str]:
        """Evaluate README for installation steps."""
        prompt = f"""
        You are an expert technical reviewer evaluating a project's README file for installation instructions.
        
        README Content:
        {readme_content}
        
        Task: Evaluate whether this README includes clear installation steps and assign a score from 1-5.
        
        Scoring Criteria:
        5 - Excellent: Clear step-by-step installation guide with all dependencies listed
        4 - Good: Installation steps present but could be more detailed
        3 - Fair: Basic installation info but missing some details
        2 - Poor: Vague or incomplete installation instructions
        1 - Very Poor: No installation instructions or completely unclear
        
        Consider:
        - Are there step-by-step installation instructions?
        - Are dependencies clearly listed?
        - Are there any prerequisites mentioned?
        - Is the installation process easy to follow?
        
        Respond in this exact format:
        Score: [1-5]
        Comments: [Your detailed explanation]
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            # Parse response
            lines = response_text.split('\n')
            score = 0
            comments = ""
            
            for line in lines:
                if line.startswith('Score:'):
                    score = int(line.split(':')[1].strip())
                elif line.startswith('Comments:'):
                    comments = line.split(':', 1)[1].strip()
            
            return score, comments
        except Exception as e:
            print(f"Error evaluating README installation: {e}")
            return 0, f"Error during evaluation: {e}"
    
    def evaluate_readme_quality(self, readme_content: str) -> Tuple[int, str]:
        """Evaluate README for grammatical quality and structure."""
        prompt = f"""
        You are an expert technical writer evaluating a project's README file for clarity, structure, and grammatical quality.
        
        README Content:
        {readme_content}
        
        Task: Rate the clarity and structure of this README on a scale from 1-5.
        
        Scoring Criteria:
        5 - Excellent: Well-structured, clear, professional writing with good formatting
        4 - Good: Generally clear with minor structural or grammatical issues
        3 - Fair: Understandable but could benefit from better organization
        2 - Poor: Unclear structure, multiple grammatical errors
        1 - Very Poor: Poorly written, difficult to understand, major issues
        
        Consider:
        - Grammar and spelling
        - Logical structure and flow
        - Use of headers and formatting
        - Clarity of explanations
        - Professional presentation
        
        Respond in this exact format:
        Score: [1-5]
        Comments: [Your detailed explanation]
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            # Parse response
            lines = response_text.split('\n')
            score = 0
            comments = ""
            
            for line in lines:
                if line.startswith('Score:'):
                    score = int(line.split(':')[1].strip())
                elif line.startswith('Comments:'):
                    comments = line.split(':', 1)[1].strip()
            
            return score, comments
        except Exception as e:
            print(f"Error evaluating README quality: {e}")
            return 0, f"Error during evaluation: {e}"
    
    def evaluate_commit_activity(self, commits: List[Dict]) -> Tuple[int, str]:
        """Evaluate commit activity during hackathon period."""
        if not commits:
            return 1, "No commits found during hackathon period"
        
        # Count commits during hackathon period
        try:
            hackathon_commits = [c for c in commits if self.hackathon_start <= c['date'] <= self.hackathon_end]
        except Exception as e:
            print(f"Error comparing commit dates: {e}")
            return 1, f"Error evaluating commit dates: {e}"
        
        prompt = f"""
        You are evaluating a project's commit activity during a hackathon period.
        
        Hackathon Period: {self.hackathon_start.strftime('%Y-%m-%d')} to {self.hackathon_end.strftime('%Y-%m-%d')}
        
        Commit Data:
        Total commits during hackathon: {len(hackathon_commits)}
        Total commits in repository: {len(commits)}
        
        Recent commit messages (last 5):
        {[c['message'][:100] + '...' if len(c['message']) > 100 else c['message'] for c in hackathon_commits[:5]]}
        
        Task: Evaluate the commit activity and assign a score from 1-5.
        
        Scoring Criteria:
        5 - Excellent: Active development with meaningful commits throughout hackathon
        4 - Good: Regular commits with good development activity
        3 - Fair: Some commits but could be more active
        2 - Poor: Very few commits or mostly minor changes
        1 - Very Poor: No commits during hackathon period
        
        Consider:
        - Number of commits during hackathon period
        - Quality of commit messages
        - Consistency of development activity
        - Whether commits show actual project development
        
        Respond in this exact format:
        Score: [1-5]
        Comments: [Your detailed explanation]
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            # Parse response
            lines = response_text.split('\n')
            score = 0
            comments = ""
            
            for line in lines:
                if line.startswith('Score:'):
                    score = int(line.split(':')[1].strip())
                elif line.startswith('Comments:'):
                    comments = line.split(':', 1)[1].strip()
            
            return score, comments
        except Exception as e:
            print(f"Error evaluating commit activity: {e}")
            return 0, f"Error during evaluation: {e}"
    
    def evaluate_project(self, repo_url: str) -> Dict:
        """Evaluate a single project and return results."""
        print(f"Evaluating: {repo_url}")
        
        try:
            owner, repo_name = self.extract_repo_info(repo_url)
            
            # Get project data
            readme_content = self.get_readme_content(owner, repo_name)
            commits = self.get_commit_history(owner, repo_name)
            
            # Evaluate README
            readme_installation_score, readme_installation_comments = self.evaluate_readme_installation(readme_content)
            readme_quality_score, readme_quality_comments = self.evaluate_readme_quality(readme_content)
            
            # Evaluate commit activity
            commit_score, commit_comments = self.evaluate_commit_activity(commits)
            
            # Calculate total score
            total_score = readme_installation_score + readme_quality_score + commit_score
            
            # Combine comments
            all_comments = f"README Installation: {readme_installation_comments} | README Quality: {readme_quality_comments} | Commit Activity: {commit_comments}"
            
            return {
                'project_name': f"{owner}/{repo_name}",
                'github_link': repo_url,
                'readme_installation_score': readme_installation_score,
                'readme_quality_score': readme_quality_score,
                'commit_activity_score': commit_score,
                'total_score': total_score,
                'comments': all_comments
            }
            
        except Exception as e:
            print(f"Error evaluating project {repo_url}: {e}")
            return {
                'project_name': repo_url,
                'github_link': repo_url,
                'readme_installation_score': 0,
                'readme_quality_score': 0,
                'commit_activity_score': 0,
                'total_score': 0,
                'comments': f"Error during evaluation: {e}"
            }
    
    def evaluate_projects_from_csv(self, input_csv_path: str, output_csv_path: str):
        """Evaluate all projects from input CSV and save results to output CSV."""
        # Read input CSV
        df = pd.read_csv(input_csv_path)
        
        if 'repo_url' not in df.columns:
            raise ValueError("Input CSV must contain a 'repo_url' column")
        
        results = []
        
        for index, row in df.iterrows():
            repo_url = row['repo_url']
            result = self.evaluate_project(repo_url)
            results.append(result)
            
            # Print progress
            print(f"Completed {index + 1}/{len(df)} projects")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")
        
        return results_df 