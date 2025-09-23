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
import time

class RateLimitError(Exception):
    """Custom exception for GitHub API rate limiting."""
    pass

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
            model_name="openai/gpt-oss-120b"
        )
        
        print("Initializing hackathon period...")
        self.hackathon_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
        self.hackathon_end = datetime(2025, 7, 21, tzinfo=timezone.utc)
        
        # Check GitHub API rate limit status
        self._check_github_rate_limit()
    
    def _check_github_rate_limit(self):
        """Check and display current GitHub API rate limit status."""
        try:
            rate_limit = self.github.get_rate_limit()
            core_limit = rate_limit.core
            search_limit = rate_limit.search
            
            print(f"GitHub API Rate Limit Status:")
            print(f"  Core API: {core_limit.remaining}/{core_limit.limit} requests remaining")
            print(f"  Search API: {search_limit.remaining}/{search_limit.limit} requests remaining")
            
            if core_limit.remaining < 100:
                print(f"  ⚠️  Warning: Low API requests remaining!")
                print(f"  ⏰  Core API resets at: {core_limit.reset}")
            
            if core_limit.remaining < 50:
                print(f"  🚨 Critical: Very low API requests remaining!")
                print(f"  Consider waiting for rate limit reset or using a different token")
                
        except Exception as e:
            print(f"Warning: Could not check GitHub API rate limit: {e}")
    
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
        """Fetch README content from GitHub repository with comprehensive search."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # First try the standard GitHub API method (root directory only)
            try:
                readme = repo.get_readme()
                content = readme.decoded_content.decode('utf-8')
                print(f"  ✓ Found README in root directory")
                return content
            except Exception:
                print(f"  → No README found in root, searching subdirectories...")
                pass
            
            # If root README not found, search common documentation locations
            common_paths = [
                'docs/README.md',
                'docs/readme.md', 
                'documentation/README.md',
                'Documentation/README.md',
                'src/README.md',
                'README/README.md',
                'doc/README.md'
            ]
            
            for path in common_paths:
                try:
                    file = repo.get_contents(path)
                    if file.type == "file":
                        content = file.decoded_content.decode('utf-8')
                        print(f"  ✓ Found README at: {path}")
                        return content
                except Exception:
                    continue
            
            # If still not found, do a comprehensive recursive search (limited depth)
            print(f"  → Performing comprehensive search...")
            content = self._search_readme_recursive(repo)
            if content:
                return content
                
            print(f"  ✗ No README file found anywhere in repository")
            return ""
            
        except Exception as e:
            print(f"Error fetching README for {owner}/{repo_name}: {e}")
            return ""
    
    def _search_readme_recursive(self, repo, path="", max_depth=3, current_depth=0) -> str:
        """Recursively search for README files with depth limit to avoid API rate limiting."""
        if current_depth >= max_depth:
            return ""
            
        try:
            contents = repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]
                
            # First pass: look for README files in current directory
            for content in contents:
                if content.type == "file" and "readme" in content.name.lower():
                    try:
                        file_content = content.decoded_content.decode('utf-8')
                        print(f"  ✓ Found README at: {content.path}")
                        return file_content
                    except Exception:
                        continue
            
            # Second pass: search subdirectories (limited depth)
            for content in contents:
                if content.type == "dir":
                    # Skip directories starting with @ or .
                    if content.name.startswith('.') or content.name.startswith('@'):
                        continue
                    
                    # Skip common directories that are unlikely to contain main README
                    skip_dirs = {
                        'lib',
                        'node_modules', 
                        'assets',
                        'static',
                        'dist',
                        'build',
                        'target',
                        '__pycache__',
                        'artifacts',
                        'vendor'
                    }
                    if content.name.lower() not in skip_dirs:
                        result = self._search_readme_recursive(repo, content.path, max_depth, current_depth + 1)
                        if result:
                            return result
                            
        except Exception as e:
            # Silently continue on errors to avoid spam, but could log for debugging
            pass
            
        return ""
    
    def get_dfx_json_content(self, owner: str, repo_name: str) -> bool:
        """Check if dfx.json file exists in the repository with comprehensive search."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # First try the standard GitHub API method (root directory only)
            try:
                dfx_file = repo.get_contents("dfx.json")
                if dfx_file.type == "file":
                    print(f"  ✓ Found dfx.json in root directory")
                    return True
            except Exception:
                print(f"  → No dfx.json found in root, searching subdirectories...")
                pass
            
            # If root dfx.json not found, search common locations
            common_paths = [
                'src/dfx.json',
                'config/dfx.json',
                'configs/dfx.json',
                'canister/dfx.json',
                'canisters/dfx.json',
                'icp/dfx.json',
                'backend/dfx.json',
                'frontend/dfx.json'
            ]
            
            for path in common_paths:
                try:
                    file = repo.get_contents(path)
                    if file.type == "file":
                        print(f"  ✓ Found dfx.json at: {path}")
                        return True
                except Exception:
                    continue
            
            # If still not found, do a comprehensive recursive search (limited depth)
            print(f"  → Performing comprehensive search...")
            found = self._search_dfx_json_recursive(repo)
            if found:
                return True
                
            print(f"  ✗ No dfx.json file found anywhere in repository")
            return False
            
        except Exception as e:
            print(f"Error searching for dfx.json in {owner}/{repo_name}: {e}")
            return False
    
    def _search_dfx_json_recursive(self, repo, path="", max_depth=3, current_depth=0) -> bool:
        """Recursively search for dfx.json files with depth limit to avoid API rate limiting."""
        if current_depth >= max_depth:
            return False
            
        try:
            contents = repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]
                
            # First pass: look for dfx.json files in current directory
            for content in contents:
                if content.type == "file" and content.name.lower() == "dfx.json":
                    print(f"  ✓ Found dfx.json at: {content.path}")
                    return True
            
            # Second pass: search subdirectories (limited depth)
            for content in contents:
                if content.type == "dir":
                    # Skip directories starting with @ or .
                    if content.name.startswith('.') or content.name.startswith('@'):
                        continue
                    
                    # Skip common directories that are unlikely to contain dfx.json
                    skip_dirs = {
                        'lib',
                        'node_modules', 
                        'assets',
                        'static',
                        'dist',
                        'build',
                        'target',
                        '__pycache__',
                        'artifacts',
                        'vendor',
                        'docs',
                        'documentation',
                        'test',
                        'tests'
                    }
                    if content.name.lower() not in skip_dirs:
                        result = self._search_dfx_json_recursive(repo, content.path, max_depth, current_depth + 1)
                        if result:
                            return True
                            
        except Exception as e:
            # Silently continue on errors to avoid spam, but could log for debugging
            pass
            
        return False

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

    def evaluate_dfx_json_presence(self, owner: str, repo_name: str) -> tuple:
        """Evaluate if dfx.json file is present in the repository."""
        has_dfx_json = self.get_dfx_json_content(owner, repo_name)
        
        if has_dfx_json:
            score = 1
            comments = "dfx.json file found - indicates ICP/Dfinity project structure"
        else:
            score = 0
            comments = "No dfx.json file found - may not be using standard ICP/Dfinity development tools"
        
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
        
        # If there are many commits, use batched approach
        if len(weekly_commits) > 5:
            return self.generate_batched_weekly_summary(owner, repo_name, weekly_commits, week_start)
        
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

    def generate_batched_weekly_summary(self, owner: str, repo_name: str, weekly_commits: list, week_start: str) -> str:
        """Generate a summary by processing commits in smaller batches to avoid context limits."""
        if not weekly_commits:
            return "No commits this week"
        
        # Process commits in batches of 5
        batch_size = 5
        summaries = []
        
        for i in range(0, len(weekly_commits), batch_size):
            batch = weekly_commits[i:i + batch_size]
            batch_summary = self.generate_weekly_summary_from_commits(batch, week_start)
            summaries.append(batch_summary)
        
        # If we have multiple batch summaries, combine them
        if len(summaries) > 1:
            combined_prompt = f"""
            You are combining multiple summaries from a development week into one cohesive summary.
            
            Week starting: {week_start}
            Number of commits: {len(weekly_commits)}
            
            Batch summaries:
            {chr(10).join([f"Batch {i+1}: {summary}" for i, summary in enumerate(summaries)])}
            
            Task: Combine these batch summaries into one concise summary (max 3 sentences) of what was built or improved this week.
            Focus on:
            - New features added
            - Existing features modified or improved
            - The overall impact of the changes
            
            Respond with only the combined summary:
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=combined_prompt)])
                combined_summary = response.content.strip()
                return combined_summary if combined_summary else summaries[0]  # Fallback to first batch summary
            except Exception as e:
                print(f"Error combining batch summaries: {e}")
                return summaries[0]  # Fallback to first batch summary
        else:
            return summaries[0]

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

    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if the error is a rate limiting error."""
        rate_limit_indicators = [
            "401", "Bad credentials",
            "403", "rate limit",
            "rate limit exceeded",
            "API rate limit",
            "too many requests",
            "quota exceeded"
        ]
        
        error_lower = error_msg.lower()
        return any(indicator.lower() in error_lower for indicator in rate_limit_indicators)
    
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
            
            # Evaluate dfx.json presence
            dfx_json_score, dfx_json_comments = self.evaluate_dfx_json_presence(owner, repo_name)
            
            # Calculate total score (readme_documentation + commit_activity + dfx_json)
            total_score = readme_documentation_score + commit_score + dfx_json_score
            
            return {
                'project_name': f"{owner}/{repo_name}",
                'github_link': repo_url,
                'readme_documentation_score': readme_documentation_score,
                'commit_activity_score': commit_score,
                'dfx_json_score': dfx_json_score,
                'total_score': total_score,
                'readme_documentation_comments': readme_documentation_comments,
                'commit_activity_comments': commit_comments,
                'dfx_json_comments': dfx_json_comments
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for rate limiting errors
            if self._is_rate_limit_error(error_msg):
                raise RateLimitError(f"GitHub API rate limit exceeded for {repo_url}: {error_msg}")
            
            print(f"Error evaluating project {repo_url}: {e}")
            return {
                'project_name': repo_url,
                'github_link': repo_url,
                'readme_documentation_score': 0,
                'commit_activity_score': 0,
                'dfx_json_score': 0,
                'total_score': 0,
                'readme_documentation_comments': f"Error during evaluation: {e}",
                'commit_activity_comments': f"Error during evaluation: {e}",
                'dfx_json_comments': f"Error during evaluation: {e}"
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
    
    def evaluate_projects_in_batches(self, input_csv_path: str, output_dir: str, batch_percentage: float = 10.0, 
                                   generate_report: bool = True, resume_from: str = None):
        """
        Evaluate projects in batches and save results incrementally.
        
        Args:
            input_csv_path: Path to input CSV file
            output_dir: Directory to save batch results
            batch_percentage: Percentage of total projects to process per batch (default: 10%)
            generate_report: Whether to generate detailed reports
            resume_from: Path to resume from a previous batch (optional)
        
        Returns:
            Path to final combined results file
        """
        print('Reading input CSV...')
        df = pd.read_csv(input_csv_path)
        
        if 'repo_url' not in df.columns:
            raise ValueError("Input CSV must contain a 'repo_url' column")
        
        total_projects = len(df)
        batch_size = max(1, int(total_projects * (batch_percentage / 100.0)))
        
        print(f"Total projects: {total_projects}")
        print(f"Batch size: {batch_size} projects ({batch_percentage}% of total)")
        print(f"Number of batches: {(total_projects + batch_size - 1) // batch_size}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize or load existing results
        all_results = []
        processed_count = 0
        
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from previous results: {resume_from}")
            existing_df = pd.read_csv(resume_from)
            all_results = existing_df.to_dict('records')
            processed_count = len(all_results)
            print(f"Already processed: {processed_count} projects")
        
        # Process projects in batches
        for batch_num in range(processed_count // batch_size, (total_projects + batch_size - 1) // batch_size):
            try:
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_projects)
                
                print(f"\n{'='*60}")
                print(f"Processing Batch {batch_num + 1}: Projects {start_idx + 1}-{end_idx} of {total_projects}")
                print(f"Progress: {processed_count}/{total_projects} ({processed_count/total_projects*100:.1f}%)")
                print(f"{'='*60}")
                
                batch_results = []
                batch_df = df.iloc[start_idx:end_idx]
                
                for idx, row in batch_df.iterrows():
                    repo_url = row['repo_url']
                    print(f"Evaluating project {idx + 1}/{total_projects}: {repo_url}")
                    
                    try:
                        result = self.evaluate_project(repo_url)
                        batch_results.append(result)
                        processed_count += 1
                        
                        # Print progress within batch
                        batch_progress = len(batch_results)
                        print(f"  ✓ Completed {batch_progress}/{len(batch_df)} in current batch")
                        
                        # Check rate limit status every 10 projects
                        if batch_progress % 10 == 0:
                            self._check_github_rate_limit()
                        
                    except RateLimitError as e:
                        print(f"\n{'!'*80}")
                        print(f"🚨 RATE LIMIT ERROR DETECTED! 🚨")
                        print(f"{'!'*80}")
                        print(f"GitHub API rate limit exceeded for project: {repo_url}")
                        print(f"Error details: {e}")
                        print(f"\nCurrent batch progress: {len(batch_results)}/{len(batch_df)} projects completed")
                        print(f"Total progress: {processed_count}/{total_projects} projects completed")
                        print(f"\nRecommendations:")
                        print(f"1. Check your GitHub token validity and permissions")
                        print(f"2. Wait for rate limits to reset (usually 1 hour)")
                        print(f"3. Use a different GitHub token if available")
                        print(f"4. Resume processing later with: --resume-from {os.path.join(output_dir, f'combined_results_{processed_count}_projects.csv')}")
                        print(f"\n{'!'*80}")
                        
                        # Save current progress before stopping
                        if batch_results:
                            batch_df_results = pd.DataFrame(batch_results)
                            batch_filename = f"batch_{batch_num + 1:03d}_results.csv"
                            batch_path = os.path.join(output_dir, batch_filename)
                            batch_df_results.to_csv(batch_path, index=False)
                            
                            # Save combined results so far
                            all_results.extend(batch_results)
                            combined_df = pd.DataFrame(all_results)
                            combined_filename = f"combined_results_{processed_count}_projects.csv"
                            combined_path = os.path.join(output_dir, combined_filename)
                            combined_df.to_csv(combined_path, index=False)
                            
                            print(f"\nProgress saved to: {combined_path}")
                        
                        # Ask user if they want to wait for rate limit reset
                        print(f"\nWould you like to:")
                        print(f"1. Wait for rate limit reset (recommended)")
                        print(f"2. Stop processing and resume later")
                        print(f"3. Continue with limited evaluation (may fail)")
                        
                        # For now, we'll stop processing but save progress
                        # In a future version, we could add interactive waiting
                        print(f"\nProcessing stopped due to rate limiting.")
                        print(f"Progress saved. Resume later with:")
                        print(f"python main.py {input_csv_path} {output_dir} --batch-mode --resume-from {combined_path}")
                        
                        raise e  # Re-raise to stop processing
                        
                    except Exception as e:
                        print(f"  ✗ Error evaluating {repo_url}: {e}")
                        # Add error result
                        error_result = {
                            'project_name': repo_url,
                            'github_link': repo_url,
                            'readme_documentation_score': 0,
                            'commit_activity_score': 0,
                            'dfx_json_score': 0,
                            'total_score': 0,
                            'readme_documentation_comments': f"Error during evaluation: {e}",
                            'commit_activity_comments': f"Error during evaluation: {e}",
                            'dfx_json_comments': f"Error during evaluation: {e}"
                        }
                        batch_results.append(error_result)
                        processed_count += 1
                
                # Save batch results
                batch_df_results = pd.DataFrame(batch_results)
                batch_filename = f"batch_{batch_num + 1:03d}_results.csv"
                batch_path = os.path.join(output_dir, batch_filename)
                
                print(f"Saving batch results to: {batch_path}")
                batch_df_results.to_csv(batch_path, index=False)
                
                # Add to all results
                all_results.extend(batch_results)
                
                # Save combined results so far
                combined_df = pd.DataFrame(all_results)
                combined_filename = f"combined_results_{processed_count}_projects.csv"
                combined_path = os.path.join(output_dir, combined_filename)
                
                print(f"Saving combined results to: {combined_path}")
                combined_df.to_csv(combined_path, index=False)
                
                # Generate batch report if requested
                if generate_report:
                    try:
                        batch_report_path = os.path.join(output_dir, f"batch_{batch_num + 1:03d}_detailed_report.txt")
                        self.create_detailed_report(batch_df_results, batch_report_path)
                        
                        # Also update combined report
                        combined_report_path = os.path.join(output_dir, f"combined_results_{processed_count}_projects_detailed_report.txt")
                        self.create_detailed_report(combined_df, combined_report_path)
                    except Exception as e:
                        print(f"Warning: Could not generate detailed report for batch {batch_num + 1}: {e}")
                        print("Continuing with batch processing...")
                
                print(f"Batch {batch_num + 1} completed. Total processed: {processed_count}/{total_projects}")
                
            except Exception as e:
                print(f"Error processing batch {batch_num + 1}: {e}")
                print("Continuing with next batch...")
                continue
        
        # Final combined results
        final_results_df = pd.DataFrame(all_results)
        final_output_path = os.path.join(output_dir, "final_results.csv")
        
        print(f"\n{'='*60}")
        print("ALL BATCHES COMPLETED!")
        print(f"Total projects evaluated: {len(all_results)}")
        print(f"Final results saved to: {final_output_path}")
        print(f"{'='*60}")
        
        final_results_df.to_csv(final_output_path, index=False)
        
        if generate_report:
            try:
                final_report_path = os.path.join(output_dir, "final_results_detailed_report.txt")
                self.create_detailed_report(final_results_df, final_report_path)
            except Exception as e:
                print(f"Warning: Could not generate final detailed report: {e}")
        
        return final_output_path
    
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
            f.write("dfx_json_score (out of 1)\n")
            f.write("total_score (out of 9)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average README Documentation Score: {results_df['readme_documentation_score'].mean():.2f}/5\n")
            f.write(f"Average Commit Activity Score: {results_df['commit_activity_score'].mean():.2f}/3\n")
            f.write(f"Average dfx.json Score: {results_df['dfx_json_score'].mean():.2f}/1\n")
            f.write(f"Average Total Score: {results_df['total_score'].mean():.2f}/9\n\n")
            
            print('Writing summary statistics...')
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Total Score: {results_df['total_score'].mean():.2f}/9\n")
            f.write(f"Average README Documentation Score: {results_df['readme_documentation_score'].mean():.2f}/5\n")
            f.write(f"Average Commit Activity Score: {results_df['commit_activity_score'].mean():.2f}/3\n")
            f.write(f"Average dfx.json Score: {results_df['dfx_json_score'].mean():.2f}/1\n\n")
            
            print('Writing top performers...')
            top_projects = results_df.nlargest(5, 'total_score')
            f.write("TOP 5 PROJECTS BY TOTAL SCORE\n")
            f.write("-" * 40 + "\n")
            for idx, (_, project) in enumerate(top_projects.iterrows(), 1):
                f.write(f"{idx}. {project['project_name']} - Score: {project['total_score']}/9\n")
                f.write(f"   GitHub: {project['github_link']}\n")
                f.write(f"   README Documentation: {project['readme_documentation_score']}/5\n")
                f.write(f"   Commit Activity: {project['commit_activity_score']}/3\n")
                f.write(f"   dfx.json Present: {project['dfx_json_score']}/1\n\n")
            
            print('Writing detailed project evaluations...')
            f.write("DETAILED PROJECT EVALUATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (_, project) in enumerate(results_df.iterrows(), 1):
                f.write(f"PROJECT {idx}: {project['project_name']}\n")
                f.write("-" * 60 + "\n")
                f.write(f"GitHub Link: {project['github_link']}\n")
                f.write(f"Total Score: {project['total_score']}/9\n")
                f.write(f"README Documentation: {project['readme_documentation_score']}/5\n")
                f.write(f"Commit Activity: {project['commit_activity_score']}/3\n")
                f.write(f"dfx.json Present: {project['dfx_json_score']}/1\n\n")
                
                f.write("README Documentation Evaluation:\n")
                f.write(f"  {project['readme_documentation_comments']}\n\n")
                
                f.write("Commit Activity Evaluation:\n")
                f.write(f"  {project['commit_activity_comments']}\n\n")
                
                f.write("dfx.json Evaluation:\n")
                f.write(f"  {project['dfx_json_comments']}\n\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"Detailed report saved to: {report_path}") 