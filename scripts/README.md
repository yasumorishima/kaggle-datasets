# Column Description Auto-Fill Scripts

## Overview

This directory contains scripts to automatically fill column descriptions in Kaggle datasets using browser console JavaScript.

## Files

- `generate_js.py` - Python script to generate JavaScript code from markdown column description files

## Usage

### Step 1: Generate JavaScript from Markdown

```bash
python generate_js.py <markdown_file> > output.js
```

Example:
```bash
python generate_js.py ../dataset1_japanese_mlb/batting_columns.md > kaggle_batting.js
```

### Step 2: Run JavaScript in Browser Console

1. Open your Kaggle dataset edit page
2. Navigate to the Column Description section
3. Press F12 to open Developer Tools
4. Go to Console tab
5. Copy the entire content of the generated `.js` file
6. Paste into the console and press Enter
7. Review the changes and click Save

## Output Example

The script will output:
```
Found 119 columns
[OK] pitch_type
[OK] game_date
...
=== SUMMARY ===
Updated: 119
Skipped: 19
Failed: 0
```

## Notes

- The script uses React-compatible value setting to ensure changes are detected
- Already filled columns will be overwritten
- Always review changes before saving
