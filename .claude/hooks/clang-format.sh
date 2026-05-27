#!/usr/bin/env bash
# PostToolUse hook: format C/C++ files with clang-format after writes/edits.
# Receives a JSON payload on stdin; exits 0 always so it never blocks Claude.

set -euo pipefail

input=$(cat)

# Extract the file path from the tool input (field may be "file_path" or "path")
file_path=$(printf '%s' "$input" | python3 -c "
import json, sys
data = json.load(sys.stdin)
ti = data.get('tool_input', {})
print(ti.get('file_path', ti.get('path', '')))
" 2>/dev/null || true)

if [[ -z "$file_path" ]]; then
    exit 0
fi

# Resolve relative paths against the workspace root (the hook's cwd is the repo root)
if [[ "$file_path" != /* ]]; then
    file_path="$PWD/$file_path"
fi

if [[ ! -f "$file_path" ]]; then
    exit 0
fi

# Only format C/C++ source files
case "$file_path" in
    *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx)
        ;;
    *)
        exit 0
        ;;
esac

clang-format -i --style=file "$file_path"
