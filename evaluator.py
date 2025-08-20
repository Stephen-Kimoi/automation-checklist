import os
import pandas as pd
from datetime import datetime, timezone, timedelta
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
    
    def get_commit_diff(self, owner: str, repo_name: str, commit_sha: str) -> str:
        """Get the diff for a specific commit, with a limit of 2000 lines."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            commit = repo.get_commit(commit_sha)
            
            # Get the files changed in this commit
            files = commit.files
            
            # Build a summary of changes
            diff_summary = []
            total_lines = 0
            
            # Convert PaginatedList to regular list to get length
            files_list = list(files)
            print(f"  Analyzing commit {commit_sha[:8]} - {len(files_list)} files changed")
            
            for file_change in files_list:
                filename = file_change.filename
                status = file_change.status  # 'added', 'removed', 'modified', 'renamed'
                
                if status == 'added':
                    diff_summary.append(f"ADDED: {filename}")
                    if hasattr(file_change, 'additions') and hasattr(file_change, 'deletions'):
                        diff_summary.append(f"  Lines: +{file_change.additions} -{file_change.deletions}")
                elif status == 'removed':
                    diff_summary.append(f"DELETED: {filename}")
                elif status == 'modified':
                    diff_summary.append(f"MODIFIED: {filename}")
                    if hasattr(file_change, 'additions') and hasattr(file_change, 'deletions'):
                        diff_summary.append(f"  Lines: +{file_change.additions} -{file_change.deletions}")
                    
                    # Get the actual diff content (patch)
                    if hasattr(file_change, 'patch') and file_change.patch:
                        patch_content = file_change.patch
                        # Limit to first 50 lines of patch to avoid overwhelming
                        patch_lines = patch_content.split('\n')[:50]
                        diff_summary.append("  Changes:")
                        diff_summary.extend([f"    {line}" for line in patch_lines])
                        total_lines += len(patch_lines)
                        
                        if len(patch_lines) == 50:
                            diff_summary.append("    ... [patch truncated]")
                elif status == 'renamed':
                    old_filename = getattr(file_change, 'previous_filename', 'unknown')
                    diff_summary.append(f"RENAMED: {old_filename} -> {filename}")
                
                # Check if we're approaching the 2000 line limit
                if total_lines > 1800:  # Leave some buffer
                    diff_summary.append("... [diff truncated due to size limit]")
                    break
            
            result = '\n'.join(diff_summary)
            print(f"  Diff summary length: {len(result.split())} words")
            return result
            
        except Exception as e:
            print(f"Error fetching diff for commit {commit_sha}: {e}")
            return f"Error fetching diff: {e}"

    def get_weekly_file_changes(self, owner: str, repo_name: str, weekly_commits: list) -> str:
        """Get a summary of file changes for a week's worth of commits."""
        if not weekly_commits:
            return "No commits this week"
        
        weekly_changes = []
        total_diff_lines = 0
        
        for commit in weekly_commits:
            commit_sha = commit['sha']
            commit_message = commit['message']
            
            weekly_changes.append(f"\nCommit: {commit_sha[:8]} - {commit_message}")
            
            # Get the diff for this commit
            diff_content = self.get_commit_diff(owner, repo_name, commit_sha)
            
            if diff_content:
                weekly_changes.append("File Changes:")
                weekly_changes.append(diff_content)
                total_diff_lines += len(diff_content.split('\n'))
                
                # Check if we're approaching the 2000 line limit
                if total_diff_lines > 1800:
                    weekly_changes.append("\n... [weekly summary truncated due to size limit]")
                    break
        
        return '\n'.join(weekly_changes)

    def get_initial_file_state(self, owner: str, repo_name: str) -> str:
        """Get the initial state of files at the start of the hackathon period."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Get the commit at the start of the hackathon period
            commits = repo.get_commits(sha=repo.default_branch, until=self.hackathon_start)
            if commits:
                initial_commit = commits[0]  # Most recent commit before hackathon start
                
                # Get the tree of files at that commit
                tree = repo.get_git_tree(sha=initial_commit.sha, recursive=True)
                
                # Build a summary of the initial file structure
                file_summary = []
                for item in tree.tree:
                    if item.type == 'blob':  # File
                        file_summary.append(f"FILE: {item.path}")
                    elif item.type == 'tree':  # Directory
                        file_summary.append(f"DIR: {item.path}/")
                
                return f"Initial state at {initial_commit.commit.author.date.strftime('%Y-%m-%d')}:\n" + '\n'.join(file_summary[:100])  # Limit to first 100 files
            else:
                return "Could not determine initial file state"
                
        except Exception as e:
            print(f"Error getting initial file state for {owner}/{repo_name}: {e}")
            return f"Error getting initial file state: {e}"
    
    def evaluate_readme_documentation(self, readme_content: str) -> tuple:
        """Evaluate README for documentation quality including installation, setup, and general documentation."""
        chunks = self.chunk_text(readme_content)
        score = 0
        comments = ""
        
        for chunk in chunks:
            prompt = f"""
            You are an expert technical writer evaluating a project's README file for comprehensive documentation quality.
            
            README Content:
            {chunk}
            
            Task: Rate the documentation quality on a scale from 1-5.
            
            Scoring Criteria:
            5 - Excellent: Strong documentation including setup instructions (for local dev), general project description, integration guide (if applicable), and contribution guidelines
            4 - Good: Good documentation with most key elements present but could be more detailed
            3 - Fair: Basic documentation present but missing some important elements
            2 - Poor: Limited documentation with significant gaps
            1 - Very Poor: Minimal or no documentation
            
            Consider:
            - Setup instructions for local development
            - General project description
            - Integration instructions (if applicable)
            - Contribution guidelines
            - Overall clarity and structure
            - Grammar and formatting
            
            Respond in this exact format:
            Score: [1-5]
            Comments: [Your detailed explanation focusing on whether it includes setup instructions, project description, integration guide, and contribution guidelines. If you cannot assess, say so explicitly.]
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
                if comments and comments.lower() != 'no documentation assessment provided.':
                    break
            except Exception as e:
                print(f"Error evaluating README documentation: {e}")
                continue
        if not comments:
            comments = "No documentation assessment provided."
        return score, comments

    def analyze_weekly_commits(self, owner: str, repo_name: str, commits: list) -> tuple:
        """Analyze commit activity by week and generate weekly summaries."""
        if not commits:
            return 0, "No commits found during hackathon period.", []
        
        # Filter commits to hackathon period
        hackathon_commits = [c for c in commits if self.hackathon_start <= c['date'] <= self.hackathon_end]
        
        print(f"  Total commits: {len(commits)}")
        print(f"  Hackathon period: {self.hackathon_start} to {self.hackathon_end}")
        print(f"  Commits in hackathon period: {len(hackathon_commits)}")
        
        if not hackathon_commits:
            return 0, "No commits found during hackathon period.", []
        
        # Group commits by week
        weekly_commits = {}
        current_date = self.hackathon_start
        while current_date <= self.hackathon_end:
            week_start = current_date
            week_end = current_date + timedelta(days=6)
            week_key = week_start.strftime('%Y-%m-%d')
            
            week_commits = [c for c in hackathon_commits if week_start <= c['date'] <= week_end]
            weekly_commits[week_key] = week_commits
            
            if week_commits:
                print(f"  Week {week_key}: {len(week_commits)} commits")
            
            current_date += timedelta(days=7)
        
        # Calculate score based on weekly activity
        total_weeks = len(weekly_commits)
        weeks_with_commits = sum(1 for commits in weekly_commits.values() if commits)
        weeks_with_multiple_commits = sum(1 for commits in weekly_commits.values() if len(commits) >= 2)
        
        # Scoring system: 0-3 scale
        if weeks_with_commits == 0:
            score = 0
            score_description = "0 - no commits"
        elif weeks_with_multiple_commits >= total_weeks * 0.8:  # 80% of weeks have 2+ commits
            score = 3
            score_description = "3 - Commits every week"
        elif weeks_with_commits >= total_weeks * 0.5:  # 50% of weeks have commits
            score = 2
            score_description = "2 - Commits every other week"
        else:
            score = 1
            score_description = "1 - 1 or 2 commits"
        
        # Generate weekly summaries
        weekly_summaries = []
        for week_start, week_commits in weekly_commits.items():
            if week_commits:
                # Use LLM to summarize what was built/improved that week based on actual file changes
                summary = self.generate_weekly_summary(owner, repo_name, week_commits, week_start)
                weekly_summaries.append(f"Week of {week_start}: {summary}")
        
        return score, score_description, weekly_summaries
    
    def generate_weekly_summary(self, owner: str, repo_name: str, weekly_commits: list, week_start: str) -> str:
        """Generate a summary of what was built/improved in a given week based on actual file changes."""
        if not weekly_commits:
            return "No commits this week"
        
        try:
            # Get the initial file state for context
            initial_state = self.get_initial_file_state(owner, repo_name)
            
            # Get the detailed file changes for this week
            weekly_changes = self.get_weekly_file_changes(owner, repo_name, weekly_commits)
            
            # Check if the total prompt would be too long for the LLM
            total_prompt_length = len(initial_state.split()) + len(weekly_changes.split()) + 200  # 200 for prompt template
            
            if total_prompt_length > 3000:  # If too long, fall back to commit message analysis
                return self.generate_weekly_summary_from_commits(weekly_commits, week_start)
            
            prompt = f"""
            You are analyzing actual file changes from a development week to summarize what features were built or improved.
            
            Week starting: {week_start}
            
            Initial File State (before hackathon):
            {initial_state}
            
            File Changes and Diffs for this week:
            {weekly_changes}
            
            Task: Provide a concise summary (max 3 sentences) of what was built or improved this week based on the actual file changes.
            Focus on:
            - New features added
            - Existing features modified or improved
            - Files added, deleted, or renamed
            - The overall impact of the changes
            - How the changes relate to the initial state
            
            If the changes are unclear or don't show meaningful development, say "Minor updates and fixes".
            
            Respond with only the summary:
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            return summary if summary else "Minor updates and fixes"
            
        except Exception as e:
            print(f"Error generating weekly summary: {e}")
            return self.generate_weekly_summary_from_commits(weekly_commits, week_start)

    def generate_weekly_summary_from_commits(self, weekly_commits: list, week_start: str) -> str:
        """Generate a summary based on commit messages when diff analysis is too long."""
        if not weekly_commits:
            return "No commits this week"
        
        # Extract commit messages
        commit_messages = [commit['message'] for commit in weekly_commits]
        
        # Limit to last 15 commit messages to avoid token limits
        recent_messages = commit_messages[-15:]
        
        prompt = f"""
        You are analyzing commit messages from a development week to summarize what features were built or improved.
        
        Week starting: {week_start}
        Number of commits: {len(weekly_commits)}
        
        Recent commit messages:
        {chr(10).join([f"- {msg}" for msg in recent_messages])}
        
        Task: Provide a concise summary (max 3 sentences) of what was built or improved this week based on the commit messages.
        Focus on:
        - New features added
        - Existing features modified or improved
        - The overall impact of the changes
        
        If the commits are unclear or don't show meaningful development, say "Minor updates and fixes".
        
        Respond with only the summary:
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            return summary if summary else "Minor updates and fixes"
        except Exception as e:
            print(f"Error in commit message analysis: {e}")
            return "Minor updates and fixes"

    def evaluate_commit_activity(self, owner: str, repo_name: str, commits: list) -> tuple:
        """Evaluate commit activity during hackathon period with new weekly scoring system."""
        score, score_description, weekly_summaries = self.analyze_weekly_commits(owner, repo_name, commits)
        
        # Combine score description with weekly summaries
        comments = f"{score_description}. "
        if weekly_summaries:
            comments += "Weekly development summary: " + "; ".join(weekly_summaries)
        else:
            comments += "No weekly development activity to summarize."
            
        return score, comments

    def evaluate_project(self, repo_url: str) -> Dict:
        print(f"Evaluating: {repo_url}")
        
        try:
            owner, repo_name = self.extract_repo_info(repo_url)
            
            # Get project data
            readme_content = self.get_readme_content(owner, repo_name)
            commits = self.get_commit_history(owner, repo_name)
            
            # Evaluate README documentation (merged installation and quality)
            readme_documentation_score, readme_documentation_comments = self.evaluate_readme_documentation(readme_content)
            
            # Evaluate commit activity with new weekly scoring
            commit_score, commit_comments = self.evaluate_commit_activity(owner, repo_name, commits)
            
            # Calculate total score (removed candid_api_score)
            total_score = readme_documentation_score + commit_score
            
            return {
                'project_name': f"{owner}/{repo_name}",
                'github_link': repo_url,
                'readme_documentation_score': readme_documentation_score,
                'commit_activity_score': commit_score,
                'total_score': total_score,
                'readme_documentation_comments': readme_documentation_comments,
                'commit_activity_comments': commit_comments
            }
            
        except Exception as e:
            print(f"Error evaluating project {repo_url}: {e}")
            return {
                'project_name': repo_url,
                'github_link': repo_url,
                'readme_documentation_score': 0,
                'commit_activity_score': 0,
                'total_score': 0,
                'readme_documentation_comments': f"Error during evaluation: {e}",
                'commit_activity_comments': f"Error during evaluation: {e}"
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
            
            # Add scoring breakdown at the top
            f.write("SCORING BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            f.write("readme_documentation_score (out of 5)\n")
            f.write("commit_activity_score (out of 3)\n")
            f.write("total_score (out of 8)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average README Documentation Score: {results_df['readme_documentation_score'].mean():.2f}/5\n")
            f.write(f"Average Commit Activity Score: {results_df['commit_activity_score'].mean():.2f}/3\n")
            f.write(f"Average Total Score: {results_df['total_score'].mean():.2f}/8\n\n")
            
            print('Writing summary statistics...')
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Total Score: {results_df['total_score'].mean():.2f}/8\n")
            f.write(f"Average README Documentation Score: {results_df['readme_documentation_score'].mean():.2f}/5\n")
            f.write(f"Average Commit Activity Score: {results_df['commit_activity_score'].mean():.2f}/3\n\n")
            
            print('Writing top performers...')
            top_projects = results_df.nlargest(5, 'total_score')
            f.write("TOP 5 PROJECTS BY TOTAL SCORE\n")
            f.write("-" * 40 + "\n")
            for idx, (_, project) in enumerate(top_projects.iterrows(), 1):
                f.write(f"{idx}. {project['project_name']} - Score: {project['total_score']}/8\n")
                f.write(f"   GitHub: {project['github_link']}\n")
                f.write(f"   README Documentation: {project['readme_documentation_score']}/5\n")
                f.write(f"   Commit Activity: {project['commit_activity_score']}/3\n\n")
            
            print('Writing detailed project evaluations...')
            f.write("DETAILED PROJECT EVALUATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (_, project) in enumerate(results_df.iterrows(), 1):
                f.write(f"PROJECT {idx}: {project['project_name']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"GitHub Link: {project['github_link']}\n")
                f.write(f"Total Score: {project['total_score']}/8\n")
                f.write(f"README Documentation: {project['readme_documentation_score']}/5\n")
                f.write(f"Commit Activity: {project['commit_activity_score']}/3\n\n")
                
                f.write("README Documentation Evaluation:\n")
                f.write(f"  {project['readme_documentation_comments']}\n\n")
                
                f.write("Commit Activity Evaluation:\n")
                f.write(f"  {project['commit_activity_comments']}\n\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"Detailed report saved to: {report_path}") 