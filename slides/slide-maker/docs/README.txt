=================================================================================
POWERPOINT CREATION TOOLKIT - COMPLETE FILE PACKAGE
=================================================================================

This package contains EVERYTHING needed to create and edit PowerPoint 
presentations in Claude Code.

=================================================================================
INCLUDED FILES
=================================================================================

DOCUMENTATION (3 files):
├── SKILL.md              Main workflow guide for creating presentations
├── html2pptx.md          Detailed HTML to PowerPoint conversion reference
└── ooxml.md              Advanced raw XML editing for existing presentations

CORE SCRIPTS (2 files - REQUIRED for creating new presentations):
├── html2pptx.js          HTML to PowerPoint converter library
└── thumbnail.py          Visual verification tool (generates slide thumbnails)

ADVANCED EDITING SCRIPTS (6 files - for editing existing presentations):
├── inventory.py          Extract text inventory from presentations
├── replace.py            Replace text in existing presentations
├── rearrange.py          Rearrange slides in presentations
├── pack.py               Pack XML files back into PPTX
├── unpack.py             Unpack PPTX into XML files for editing
└── validate.py           Validate OOXML structure

=================================================================================
QUICK START - CREATING NEW PRESENTATIONS
=================================================================================

If you want to CREATE NEW presentations from scratch, you only need:
✓ SKILL.md (read this first)
✓ html2pptx.md (reference for HTML syntax)
✓ html2pptx.js (save to your working directory)
✓ thumbnail.py (save to your working directory)

WORKFLOW:
1. Install dependencies (see SKILL.md Section on Dependencies)
2. Create HTML files for each slide
3. Run html2pptx.js to convert to PowerPoint
4. Use thumbnail.py to visually verify slides
5. Iterate if needed

=================================================================================
ADVANCED - EDITING EXISTING PRESENTATIONS
=================================================================================

If you want to EDIT EXISTING presentations (templates, bulk text replacement):
✓ Read ooxml.md for XML structure details
✓ Use unpack.py to extract PPTX to XML
✓ Use inventory.py to see all text elements
✓ Use replace.py to bulk replace text
✓ Use pack.py to repackage XML back to PPTX

=================================================================================
DEPENDENCIES
=================================================================================

NPM PACKAGES (required):
npm install -g pptxgenjs playwright sharp react react-dom react-icons

PYTHON PACKAGES (required):
pip install matplotlib markitdown[pptx] --break-system-packages

OPTIONAL (for thumbnail generation):
sudo apt-get install libreoffice poppler-utils

=================================================================================
FILE USAGE MATRIX
=================================================================================

TASK: Create presentation from scratch
FILES NEEDED: SKILL.md, html2pptx.md, html2pptx.js, thumbnail.py
START HERE: Read SKILL.md completely

TASK: Add LaTeX formulas to slides
FILES NEEDED: SKILL.md (see LaTeX section), html2pptx.js, matplotlib
START HERE: SKILL.md Section 5

TASK: Edit existing presentation template
FILES NEEDED: ooxml.md, inventory.py, replace.py, unpack.py, pack.py
START HERE: Read ooxml.md, then SKILL.md template workflow section

TASK: Bulk replace text in presentations
FILES NEEDED: inventory.py, replace.py
START HERE: Use inventory.py to extract text, then replace.py

TASK: Debug layout issues
FILES NEEDED: thumbnail.py, html2pptx.js (error messages)
START HERE: Generate thumbnails to see visual issues

=================================================================================
TROUBLESHOOTING
=================================================================================

ISSUE: "Module not found" errors
FIX: Install dependencies listed above

ISSUE: "Content overflows body" errors
FIX: Reduce padding/font sizes in HTML, leave 0.5" bottom margin

ISSUE: Can't see slides visually
FIX: Run thumbnail.py to generate images of slides

ISSUE: Charts not appearing
FIX: Ensure placeholder divs in HTML, add charts after conversion

ISSUE: LaTeX formulas don't render
FIX: Use matplotlib to create PNG images first, reference in HTML

=================================================================================
RECOMMENDED READING ORDER
=================================================================================

FOR BEGINNERS (creating first presentation):
1. Read SKILL.md completely (20 min)
2. Skim html2pptx.md for HTML rules (10 min)
3. Follow workflow in SKILL.md Section 6
4. Create simple 3-slide test presentation
5. Generate thumbnails to verify

FOR ADVANCED USERS (editing templates):
1. Read SKILL.md template workflow section
2. Read ooxml.md for XML structure
3. Practice with unpack.py on example file
4. Use inventory.py and replace.py for bulk edits

=================================================================================
SUPPORT RESOURCES
=================================================================================

All documentation is self-contained in this package.

Key sections in SKILL.md:
- Section 1: Dependencies
- Section 4: HTML slide rules
- Section 5: LaTeX formulas
- Section 6: Complete workflow
- Section 9: Troubleshooting

Key sections in html2pptx.md:
- Creating HTML Slides
- Supported Elements
- Critical Text Rules
- Using PptxGenJS (charts/tables)

Key sections in ooxml.md:
- Presentation Structure
- Text Box XML
- Images and Media
- Validation rules

=================================================================================
END OF README
=================================================================================
