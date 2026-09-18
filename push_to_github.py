"""
GitHub Push Script for Hiver SDE Intern Assignment
Run this from the Hiver project folder: python push_to_github.py
It will guide you step-by-step.
"""
import os, subprocess, sys

def run(cmd, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print("STDERR:", result.stderr.strip())
    if check and result.returncode != 0:
        print(f"ERROR: Command failed (code {result.returncode})")
    return result

print("=" * 60)
print("  HIVER ASSIGNMENT — GITHUB PUSH GUIDE")
print("=" * 60)

print("""
BEFORE RUNNING THIS SCRIPT:
1. Go to https://github.com/new
2. Create a NEW public repository named: hiver-sde-intern-assignment
3. Do NOT initialize with README or .gitignore (keep it empty)
4. Copy the repo URL (e.g., https://github.com/YOUR_NAME/hiver-sde-intern-assignment.git)
""")

repo_url = input("Paste your GitHub repo URL here: ").strip()
if not repo_url:
    print("No URL provided. Exiting.")
    sys.exit(1)

print("\nInitializing git repository...")
run("git init")
run('git config user.email "you@example.com"')
run('git config user.name "Hiver Intern"')

print("\nAdding all files (large data files are excluded by .gitignore)...")
run("git add .")

print("\nCreating initial commit...")
run('git commit -m "Hiver SDE Intern: AmazonHelp AI customer-support agent"')

print("\nRenaming branch to main...")
run("git branch -M main")

print(f"\nAdding remote origin: {repo_url}")
run(f"git remote add origin {repo_url}")

print("\nPushing to GitHub...")
result = run(f"git push -u origin main", check=False)

if result.returncode == 0:
    print("\n" + "=" * 60)
    print("SUCCESS! Your code is on GitHub.")
    print(f"\nRepo URL: {repo_url.replace('.git', '')}")
    print("\nNEXT STEPS:")
    print("1. Open your repo URL in a browser to verify all files are there")
    print("2. Submit via the Notion form:")
    print("   https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f")
    print("3. Paste your GitHub repo URL in the submission form")
    print("4. Upload / link report/report.md")
    print("=" * 60)
else:
    print("\nPush failed. Common fixes:")
    print("- Make sure you're logged into GitHub (run: git credential-store)")
    print("- Or use GitHub Desktop to push instead")
    print("- Or generate a Personal Access Token at:")
    print("  https://github.com/settings/tokens")
    print("  Then use: git remote set-url origin https://TOKEN@github.com/YOUR/repo.git")
