import os
from github import Github
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

def test_commit_history(repo_url, hackathon_start, hackathon_end):
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print('GITHUB_TOKEN not set in environment!')
        return
    g = Github(github_token)
    print("Parsing owner/repo...")
    if repo_url.endswith('/'):
        repo_url = repo_url[:-1]
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    print("Splitting owner/repo...")
    parts = repo_url.split('/')
    owner = parts[-2]
    repo_name = parts[-1]
    print(f'Owner: {owner}, Repo: {repo_name}')
    print("Getting repo...")
    repo = g.get_repo(f"{owner}/{repo_name}")
    print("Getting default branch...")
    branch = repo.default_branch
    print(f'Default branch: {branch}')
    print(f'Hackathon period: {hackathon_start} to {hackathon_end}')
    commits = repo.get_commits(sha=branch, since=hackathon_start, until=hackathon_end)
    commit_list = list(commits)
    print(f"Total commits fetched in period: {len(commit_list)}")
    for c in commit_list:
        commit_date = c.commit.author.date
        print(f"Commit: {c.sha[:7]} | Date: {commit_date} | Message: {c.commit.message[:60]}")
    if not commit_list:
        print('No commits found in the specified period.')

if __name__ == "__main__":
    repo_url = "https://github.com/franRappazzini/PayPeer"
    hackathon_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    hackathon_end = datetime(2025, 7, 21, tzinfo=timezone.utc)
    test_commit_history(repo_url, hackathon_start, hackathon_end) 