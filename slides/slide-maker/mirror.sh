#!/bin/bash
# Mirror documentation files to claude.md and AGENTS.md

if [ ! -d docs ]; then
    echo "Error: docs/ directory not found"
    exit 1
fi

# Concatenate all docs into claude.md
cat docs/preamble.md docs/html2pptx.md docs/ooxml.md docs/SKILL.md > claude.md

# Mirror to AGENTS.md
cp claude.md AGENTS.md

echo "✓ Mirrored docs/ → claude.md → AGENTS.md"
