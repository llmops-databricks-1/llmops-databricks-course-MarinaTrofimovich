# Working with Git Remotes

This repo uses multiple Git remotes. Each remote points to a different GitHub repository.

## Current Remotes

| Remote    | URL                                                                                  | Purpose                        |
|-----------|--------------------------------------------------------------------------------------|--------------------------------|
| `origin`  | https://github.com/llmops-databricks-1/course-code-hub.git                          | Upstream course repository     |
| `laya`    | git@github.com:layagroup/course-code-hub.git                                        | Internal team repository       |
| `student` | https://github.com/llmops-databricks-1/llmops-databricks-course-MarinaTrofimovich.git | Personal student submissions   |

## View Remotes

```bash
git remote -v
```

## Add a New Remote

```bash
git remote add <name> <url>
```

Example:

```bash
git remote add student https://github.com/llmops-databricks-1/llmops-databricks-course-MarinaTrofimovich.git
```

## Push to a Specific Remote

Push the current branch:

```bash
git push <remote> <branch>
```

Examples:

```bash
# Push week2 branch to student repo
git push student week2

# Push week2 branch to internal team repo
git push laya week2

# Push to upstream course repo (use with caution)
git push origin week2
```

## Push a New Branch

When you push a branch that does not exist on the remote yet, Git creates it automatically:

```bash
git push student week3
```

## Pull from a Specific Remote

```bash
git pull <remote> <branch>
```

Example:

```bash
# Pull latest changes from the upstream course repo
git pull origin main
```

## Typical Weekly Workflow

1. Make sure you are on the correct branch:

   ```bash
   git checkout week2
   ```

2. Stage and commit your changes:

   ```bash
   git add -A
   git commit -m "week2 submission"
   ```

3. Push to your student repo:

   ```bash
   git push student week2
   ```

4. For the next week, create a new branch from the current one:

   ```bash
   git checkout -b week3
   # ... do your work ...
   git add -A
   git commit -m "week3 submission"
   git push student week3
   ```

## Rename a Remote

```bash
git remote rename <old-name> <new-name>
```

## Remove a Remote

```bash
git remote remove <name>
```
