# SLIDE MAKER - MANDATORY WORKFLOW

## CRITICAL INSTRUCTIONS

**YOU MUST USE THE INSTRUCTIONS IN THIS FILE (claude.md / AGENTS.md) AND ALL TOOLS IN THIS PROJECT TO CREATE SLIDES.**

**NO OTHER WAY OF CREATING SLIDES IS ALLOWED.**

### Available Tools for Slide Creation

1. **scripts/html2pptx.js** - Core library to convert HTML slides to PowerPoint
2. **scripts/thumbnail.py** - Generate visual thumbnail grids for verification
3. **scripts/inventory.py** - Extract text inventory from existing presentations
4. **scripts/replace.py** - Bulk replace text in presentations
5. **scripts/rearrange.py** - Duplicate, reorder, and delete slides
6. **scripts/pack.py** - Pack XML files into PPTX
7. **scripts/unpack.py** - Unpack PPTX into XML files
8. **scripts/validate.py** - Validate OOXML structure

### Workflow Requirements

- **For new presentations**: Use scripts/html2pptx.js workflow (HTML → PowerPoint)
- **For editing existing presentations**: Use scripts/unpack.py → edit XML → scripts/pack.py workflow
- **For template-based presentations**: Use scripts/rearrange.py → scripts/inventory.py → scripts/replace.py workflow
- **Always validate visually**: Use scripts/thumbnail.py to generate thumbnails and inspect slides
- **Output location**: Create all slides in workspace/presentations/

### Mirroring Rule

**IMPORTANT**: claude.md and AGENTS.md must always mirror each other exactly.

When making updates:
```bash
./mirror.sh  # Automatically sync docs/ → claude.md → AGENTS.md
```

### Repository Structure

```
slide-maker/
├── scripts/              # All executable tools
│   ├── html2pptx.js     # HTML → PPTX converter
│   ├── inventory.py     # Extract text from PPTX
│   ├── replace.py       # Bulk text replacement
│   ├── rearrange.py     # Reorder/duplicate slides
│   ├── pack.py          # XML → PPTX
│   ├── unpack.py        # PPTX → XML
│   ├── validate.py      # OOXML validation
│   └── thumbnail.py     # Generate thumbnails
├── docs/                 # Source documentation
│   ├── preamble.md      # This workflow header
│   ├── html2pptx.md     # HTML workflow guide
│   ├── ooxml.md         # OOXML reference
│   └── SKILL.md         # PowerPoint suite guide
├── workspace/            # Work area (gitignored)
│   ├── presentations/   # Final .pptx files
│   ├── html-slides/     # HTML sources
│   ├── images/          # Generated assets
│   └── temp/            # Temporary files
├── claude.md             # Generated: concatenated instructions
├── AGENTS.md             # Mirror of claude.md
└── mirror.sh             # Sync script
```

**Default working locations**:
- **HTML slides**: `workspace/html-slides/slide-{N}.html`
- **Final presentations**: `workspace/presentations/{project-name}.pptx`
- **Generated assets**: `workspace/images/{asset-name}.png`
- **Temporary work**: `workspace/temp/` for unpacking, validation, etc.

**Usage examples**:
```javascript
// In presentation generation script
const html2pptx = require('./scripts/html2pptx');
const { slide } = await html2pptx('workspace/html-slides/slide-1.html', pptx);
await pptx.writeFile('workspace/presentations/my-presentation.pptx');
```

```bash
# Script usage
python scripts/thumbnail.py workspace/presentations/output.pptx
python scripts/inventory.py template.pptx workspace/temp/inventory.json
python scripts/replace.py working.pptx replacements.json output.pptx
```

---

