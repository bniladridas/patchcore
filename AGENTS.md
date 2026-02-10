# Agent Guidelines

## Commit Author Configuration

When working with this repository, configure git commit authors as follows:

### Primary Author (Author field)
- Name: jupyter[ml]
- Email: jupyter-ml-bot@users.noreply.github.com

### Secondary Author (Co-author only)
- Name: Niladri Das
- Email: bniladridas@users.noreply.github.com

### Setting Up Git Config

```bash
# Set author (jupyter[ml]) for commits
git config user.name "jupyter[ml]"
git config user.email "jupyter-ml-bot@users.noreply.github.com"
```

### Creating Co-Authored Commits

Always use this format for commits to have both authors shown on GitHub:

```bash
git commit -m "commit message

Co-authored-by: jupyter[ml] <jupyter-ml-bot@users.noreply.github.com>
Co-authored-by: Niladri Das <bniladridas@users.noreply.github.com>"
```

### Amending Commits with Author Changes

```bash
# Change author to jupyter[ml]
git commit --amend --author="jupyter[ml] <jupyter-ml-bot@users.noreply.github.com>" --no-edit

# Force push after amend
git push --force
```

### GitHub Commit Display

With this setup:
- **Author field**: jupyter[ml] <jupyter-ml-bot@users.noreply.github.com>
- **Committer field**: jupyter[ml] <jupyter-ml-bot@users.noreply.github.com>
- **Co-authored-by lines**: Both jupyter[ml] and Niladri Das appear in the commit message

Note: The actual committer in git is determined by the user who pushes, not by text in commit messages.
