#!/bin/bash
set -ex
export PATH="$HOME/Library/pnpm:$PATH"
cd "$(dirname "$0")/.."

cd frontend
pnpm install
# pnpm env use --global lts
pnpm dev
