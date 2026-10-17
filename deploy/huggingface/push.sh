#!/usr/bin/env bash
# Publish the current commit to a Hugging Face Docker Space.
# 1. Create an empty Space at huggingface.co (SDK: Docker).
# 2. Log in once (`hf auth login`, or use a write token as the git password).
# 3. Run: deploy/huggingface/push.sh <hf-username>/<space-name>
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <hf-username>/<space-name>"; exit 1; }
root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"
git -C "$root" archive HEAD | tar -x -C "$tmp"
cp "$root/deploy/huggingface/README.md" "$tmp/README.md"
cd "$tmp"
git init -q -b main
git add -A
git commit -q -m "Deploy from d4mz3y/phishing-detection-app"
git push --force "https://huggingface.co/spaces/$1" main
echo "Pushed. Build logs: https://huggingface.co/spaces/$1"
