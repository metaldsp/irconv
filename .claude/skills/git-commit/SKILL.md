---
name: git-commit
description: >
  Use when creating Git commits to ensure they follow the Conventional Commits specification. It provides guidance on commit message structure, types, scopes, and best practices for writing clear, consistent, and automated-friendly commit messages. Use when committing code changes or reviewing commit history.
allowed-tools: Bash
---

# Conventional Commits

Create standardized git commits per the Conventional Commits 1.0.0 spec.

## Format

```
<type>[(<scope>)][!]: <description>

[body]

[footer(s)]
```

- **type**: noun (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`). Case-insensitive.
- **scope**: optional noun describing the affected area, e.g. `feat(parser):`.
- **description**: short summary immediately after `:<space>`. Imperative mood, <72 chars.
- **body**: optional; begins one blank line after description. Free-form paragraphs; wrap lines at 72 chars.
- **footers**: optional; begin one blank line after body. Format: `Token: value` or `Token #value`. Token uses `-` instead of spaces (e.g. `Acked-by`). Parsing stops when the next valid token/separator pair is found.
- **AI assistance**: when a commit was AI-assisted, add `Assisted-by: AGENT_NAME:MODEL_VERSION` (e.g. `Assisted-by: GitHub Copilot:claude-sonnet-4-6`, `Assisted-by: Claude Code:claude-sonnet-4-6`).

## SemVer Mapping

| Commit | SemVer bump |
|--------|-------------|
| `fix` | PATCH |
| `feat` | MINOR |
| `BREAKING CHANGE` (any type) | MAJOR |

## Breaking Changes

Either append `!` before `:` (any type/scope), or add a `BREAKING CHANGE:` footer — or both:

```
feat(api)!: drop XML support

BREAKING CHANGE: XML endpoints removed; use JSON equivalents.
```

`BREAKING-CHANGE` is synonymous with `BREAKING CHANGE` as a footer token.

## Revert

Use type `revert` and reference reverted SHAs in a `Refs` footer:

```
revert: let us never speak of the noodle incident

Refs: 676104e, a215868
```

## Workflow

1. **Check diff/status**
   ```bash
   git status --porcelain
   git diff --staged   # or: git diff (if nothing staged)
   ```

2. **Stage files** (if needed)
   ```bash
   git add <files>   # or: git add -p  (interactive)
   ```
   Never commit secrets (`.env`, credentials, private keys).

3. **Determine** type, scope, description from the diff. One logical change per commit.

4. **Commit**
   ```bash
   git commit -m "<type>[(<scope>)][!]: <description>"
   ```
   For multi-line messages use `-m` multiple times or an editor.

## Git Safety

- NEVER update git config
- NEVER use `--force` or hard reset without explicit request
- NEVER use `--no-verify` unless asked
- NEVER force-push `main`/`master`
- If a hook rejects a commit, fix the issue and create a NEW commit (don't amend)
