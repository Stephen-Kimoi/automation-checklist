import os
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from github import Github
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import re
import glob

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
        
        print("Initializing hackathon period...")
        self.hackathon_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
        self.hackathon_end = datetime(2025, 7, 21, tzinfo=timezone.utc)
    
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
    
    def extract_installation_section(self, readme_content: str) -> str:
        """Extract the Installation section from the README using regex."""
        match = re.search(r'(#+\s*Installation[\s\S]+?)(?=\n#+|$)', readme_content, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def chunk_text(self, text: str, chunk_size: int = 3500, overlap: int = 500):
        """Split text into overlapping chunks for LLM processing."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_readme_content(self, owner: str, repo_name: str) -> str:
        """Fetch README content from GitHub repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            readme = repo.get_readme()
            content = readme.decoded_content.decode('utf-8')
            return content
        except Exception as e:
            print(f"Error fetching README for {owner}/{repo_name}: {e}")
            return ""
    
    def get_commit_history(self, owner: str, repo_name: str) -> list:
        """Fetch commit history during hackathon period from the default branch only."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            branch = repo.default_branch
            commits = repo.get_commits(sha=branch, since=self.hackathon_start, until=self.hackathon_end)
            commit_data = []
            for commit in commits:
                commit_date = commit.commit.author.date 
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
    
    def evaluate_readme_quality(self, readme_content: str) -> tuple:
        """Evaluate README for grammatical quality and structure, using chunking if needed."""
        chunks = self.chunk_text(readme_content)
        score = 0
        comments = ""
        for chunk in chunks:
            prompt = f"""
            You are an expert technical writer evaluating a project's README file for clarity, structure, and grammatical quality.
            
            README Content:
            {chunk}
            
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
            Comments: [Your detailed explanation. If you cannot assess, say so explicitly.]
            """
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content
                lines = response_text.split('\n')
                for line in lines:
                    if line.startswith('Score:'):
                        try:
                            score = int(line.split(':')[1].strip())
                        except Exception:
                            score = 0
                    elif line.startswith('Comments:'):
                        comments = line.split(':', 1)[1].strip()
                if comments and comments.lower() != 'no quality assessment provided.':
                    break
            except Exception as e:
                print(f"Error evaluating README quality: {e}")
                continue
        if not comments:
            comments = "No quality assessment provided."
        return score, comments

    def evaluate_readme_installation(self, readme_content: str) -> tuple:
        """Evaluate README for installation steps, using section extraction and chunking."""
        section = self.extract_installation_section(readme_content)
        if section:
            content_to_evaluate = section
        else:
            # Fallback: chunk README and search for installation steps in each chunk
            chunks = self.chunk_text(readme_content)
            found = None
            for chunk in chunks:
                if re.search(r'install|setup|getting started', chunk, re.IGNORECASE):
                    found = chunk
                    break
            content_to_evaluate = found if found else readme_content[:3500]
        prompt = f"""
        You are an expert technical reviewer evaluating a project's README file for installation instructions.
        
        README Content:
        {content_to_evaluate}
        
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
        
        If the README provides a basic overview of how to run the project, that is sufficient for a score of 3 or above. Do not be overly critical if the basics are present.
        
        Respond in this exact format:
        Score: [1-5]
        Comments: [Your detailed explanation. If there are no installation steps, say so explicitly.]
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
                    try:
                        score = int(line.split(':')[1].strip())
                    except Exception:
                        score = 0
                elif line.startswith('Comments:'):
                    comments = line.split(':', 1)[1].strip()
            # Post-process: If basics are present, remove excessive nitpicking
            if score >= 3 and 'missing' in comments.lower() and 'basic' in comments.lower():
                comments = "The README provides a basic overview of how to run the project, which is sufficient."
            if not comments:
                comments = "No installation steps found."
            return score, comments
        except Exception as e:
            print(f"Error evaluating README installation: {e}")
            return 0, f"Error during evaluation: {e}"

    def evaluate_commit_activity(self, commits: list) -> tuple:
        """Evaluate commit activity during hackathon period."""
        if not commits:
            return 1, "No commits found during hackathon period."
        # Only use commits in the hackathon period
        hackathon_commits = [c for c in commits if self.hackathon_start <= c['date'] <= self.hackathon_end]
        if not hackathon_commits:
            return 1, "No commits found during hackathon period."
        prompt = f"""
        You are evaluating a project's commit activity during a hackathon period.
        
        Hackathon Period: {self.hackathon_start.strftime('%Y-%m-%d')} to {self.hackathon_end.strftime('%Y-%m-%d')}
        
        Commit Data:
        Total commits during hackathon: {len(hackathon_commits)}
        
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
        
        Respond in this exact format:
        Score: [1-5]
        Comments: [Your detailed explanation. If there are no commits, say so explicitly.]
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
                    try:
                        score = int(line.split(':')[1].strip())
                    except Exception:
                        score = 0
                elif line.startswith('Comments:'):
                    comments = line.split(':', 1)[1].strip()
            if not comments:
                comments = "No commits during hackathon period."
            return score, comments
        except Exception as e:
            print(f"Error evaluating commit activity: {e}")
            return 0, f"Error during evaluation: {e}"
    
    def find_candid_file(self, owner: str, repo_name: str) -> str:
        """Try to find a .did file in the repo using the GitHub API."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            contents = repo.get_contents("")
            # Search root and common subdirs
            candidates = [c for c in contents if c.name.endswith('.did')]
            if not candidates:
                # Check common subdirs
                for subdir in ['backend', 'canisters', 'src']:
                    try:
                        sub_contents = repo.get_contents(subdir)
                        candidates += [c for c in sub_contents if c.name.endswith('.did')]
                    except Exception:
                        continue
            if candidates:
                # Return the content of the first .did file found
                did_file = candidates[0]
                return did_file.decoded_content.decode('utf-8')
            else:
                return None
        except Exception as e:
            print(f"Error finding Candid file for {owner}/{repo_name}: {e}")
            return None

    def evaluate_candid_interface_score_and_short(self, candid_content: str, filename: str) -> tuple:
        """Short evaluation of a Candid interface using the LLM, returning a score and a short comment."""
        if not candid_content:
            return 0, f"File: {filename}\nScore: 0\nComment: No Candid (.did) file content found."
        # Truncate very large Candid files to avoid token limits
        max_content_length = 2000
        if len(candid_content) > max_content_length:
            candid_content = candid_content[:max_content_length] + "\n\n[Content truncated due to size]"
        prompt = f"""
        You are an expert ICP developer. Review the following Candid interface and provide:
        - A score from 1-5 (see rubric below)
        - A short statement (1-2 sentences) on whether it looks reasonable and follows best practices.

        Candid Interface:
        {candid_content}

        Rubric:
        5 - Excellent: Clear, well-documented, follows best practices
        4 - Good: Generally well-designed with minor issues
        3 - Fair: Functional but could be improved
        2 - Poor: Has significant design issues
        1 - Very Poor: Poorly designed or incomplete

        Respond in this format:
        File: {filename}
        Score: <number 1-5>
        Comment: <your short statement>
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            score = 0
            comment = ""
            # Parse score and comment from response
            for line in response_text.split('\n'):
                if line.lower().startswith('score:'):
                    try:
                        score = int(line.split(':', 1)[1].strip())
                    except Exception:
                        score = 0
                elif line.lower().startswith('comment:'):
                    comment = line.split(':', 1)[1].strip()
            # Fallback if LLM doesn't follow format
            if not comment:
                comment = response_text
            formatted = f"File: {filename}\nScore: {score}\nComment: {comment}"
            return score, formatted
        except Exception as e:
            print(f"Error evaluating Candid interface (score+short) for {filename}: {e}")
            return 0, f"File: {filename}\nScore: 0\nComment: Error during evaluation: {e}"

    def find_all_candid_files(self, owner: str, repo_name: str) -> list:
        """Recursively find all .did files in the repo using the GitHub API."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            branch = repo.default_branch
            tree = repo.get_git_tree(sha=branch, recursive=True)
            did_files = [f.path for f in tree.tree if f.path.endswith('.did')]
            return did_files
        except Exception as e:
            print(f"Error finding Candid files for {owner}/{repo_name}: {e}")
            return []

    def get_candid_file_content(self, repo, path: str) -> str:
        try:
            file_content = repo.get_contents(path)
            return file_content.decoded_content.decode('utf-8')
        except Exception as e:
            print(f"Error fetching Candid file content at {path}: {e}")
            return None

    def evaluate_all_candid_interfaces(self, owner: str, repo_name: str) -> tuple:
        """Find and evaluate all .did files, return average score and all short comments as a single string."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            did_files = self.find_all_candid_files(owner, repo_name)
            if not did_files:
                return 0, "No Candid (.did) file found in the repository."
            scores = []
            comments = []
            for path in did_files:
                content = self.get_candid_file_content(repo, path)
                score, comment = self.evaluate_candid_interface_score_and_short(content, path)
                scores.append(score)
                comments.append(comment)
            avg_score = round(sum(scores) / len(scores), 2) if scores else 0
            all_comments = "\n\n".join(comments)
            return avg_score, all_comments
        except Exception as e:
            print(f"Error evaluating all Candid interfaces: {e}")
            return 0, f"Error during evaluation: {e}"

    def evaluate_project(self, repo_url: str) -> Dict:
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
            
            # Evaluate all Candid interfaces
            candid_api_score, candid_api_comments = self.evaluate_all_candid_interfaces(owner, repo_name)
            
            # Calculate total score (add candid_api_score)
            total_score = readme_installation_score + readme_quality_score + commit_score + candid_api_score
            
            return {
                'project_name': f"{owner}/{repo_name}",
                'github_link': repo_url,
                'readme_installation_score': readme_installation_score,
                'readme_quality_score': readme_quality_score,
                'commit_activity_score': commit_score,
                'candid_api_score': candid_api_score,
                'total_score': total_score,
                'readme_installation_comments': readme_installation_comments,
                'readme_quality_comments': readme_quality_comments,
                'commit_activity_comments': commit_comments,
                'candid_api_comments': candid_api_comments
            }
            
        except Exception as e:
            print(f"Error evaluating project {repo_url}: {e}")
            return {
                'project_name': repo_url,
                'github_link': repo_url,
                'readme_installation_score': 0,
                'readme_quality_score': 0,
                'commit_activity_score': 0,
                'candid_api_score': 0,
                'total_score': 0,
                'readme_installation_comments': f"Error during evaluation: {e}",
                'readme_quality_comments': f"Error during evaluation: {e}",
                'commit_activity_comments': f"Error during evaluation: {e}",
                'candid_api_comments': f"Error during evaluation: {e}"
            }
    
    def evaluate_projects_from_csv(self, input_csv_path: str, output_csv_path: str, generate_report: bool = True):
        """Evaluate all projects from input CSV and save results to output CSV."""
        print('Reading input CSV...')
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
        
        print('Creating results DataFrame...')
        results_df = pd.DataFrame(results)
        
        print('Saving results to CSV...')
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")
        
        print('Generating detailed report...')
        if generate_report:
            report_path = output_csv_path.replace('.csv', '_detailed_report.txt')
            self.create_detailed_report(results_df, report_path)
        
        return results_df
    
    def create_detailed_report(self, results_df: pd.DataFrame, report_path: str):
        """Create a detailed, readable report from evaluation results."""
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ICP PROJECT EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Hackathon Period: {self.hackathon_start.strftime('%Y-%m-%d')} to {self.hackathon_end.strftime('%Y-%m-%d')}\n")
            f.write(f"Total Projects Evaluated: {len(results_df)}\n\n")
            
            print('Writing summary statistics...')
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Total Score: {results_df['total_score'].mean():.2f}/15\n")
            f.write(f"Average README Installation Score: {results_df['readme_installation_score'].mean():.2f}/5\n")
            f.write(f"Average README Quality Score: {results_df['readme_quality_score'].mean():.2f}/5\n")
            f.write(f"Average Commit Activity Score: {results_df['commit_activity_score'].mean():.2f}/5\n")
            f.write(f"Average Candid API Score: {results_df['candid_api_score'].mean():.2f}/5\n\n")
            
            print('Writing top performers...')
            top_projects = results_df.nlargest(5, 'total_score')
            f.write("TOP 5 PROJECTS BY TOTAL SCORE\n")
            f.write("-" * 40 + "\n")
            for idx, (_, project) in enumerate(top_projects.iterrows(), 1):
                f.write(f"{idx}. {project['project_name']} - Score: {project['total_score']}/15\n")
                f.write(f"   GitHub: {project['github_link']}\n")
                f.write(f"   README Installation: {project['readme_installation_score']}/5\n")
                f.write(f"   README Quality: {project['readme_quality_score']}/5\n")
                f.write(f"   Commit Activity: {project['commit_activity_score']}/5\n")
                f.write(f"   Candid API: {project['candid_api_score']}/5\n\n")
            
            print('Writing detailed project evaluations...')
            f.write("DETAILED PROJECT EVALUATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (_, project) in enumerate(results_df.iterrows(), 1):
                f.write(f"PROJECT {idx}: {project['project_name']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"GitHub Link: {project['github_link']}\n")
                f.write(f"Total Score: {project['total_score']}/15\n")
                f.write(f"README Installation: {project['readme_installation_score']}/5\n")
                f.write(f"README Quality: {project['readme_quality_score']}/5\n")
                f.write(f"Commit Activity: {project['commit_activity_score']}/5\n")
                f.write(f"Candid API: {project['candid_api_score']}/5\n\n")
                
                print('Parsing and formatting comments...')
                comments = project['readme_installation_comments']
                if 'README Installation:' in comments:
                    parts = comments.split('README Installation:')
                    if len(parts) > 1:
                        readme_install = parts[1].split('README Quality:')[0].strip()
                        f.write("README Installation Evaluation:\n")
                        f.write(f"  {readme_install}\n\n")
                
                if 'README Quality:' in comments:
                    parts = comments.split('README Quality:')
                    if len(parts) > 1:
                        readme_quality = parts[1].split('Commit Activity:')[0].strip()
                        f.write("README Quality Evaluation:\n")
                        f.write(f"  {readme_quality}\n\n")
                
                if 'Commit Activity:' in comments:
                    parts = comments.split('Commit Activity:')
                    if len(parts) > 1:
                        commit_activity = parts[1].strip()
                        f.write("Commit Activity Evaluation:\n")
                        f.write(f"  {commit_activity}\n\n")
                
                if 'Candid API:' in comments:
                    parts = comments.split('Candid API:')
                    if len(parts) > 1:
                        candid_api = parts[1].strip()
                        f.write("Candid API Evaluation:\n")
                        f.write(f"  {candid_api}\n\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"Detailed report saved to: {report_path}") 