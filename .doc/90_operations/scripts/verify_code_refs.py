#!/usr/bin/env python3
"""
Verify Code References in Documentation

Scans all markdown files in .doc/ for code references in the format:
  path/to/file.ext:line
  path/to/file.ext:start-end

Validates that:
1. Referenced files exist
2. Line numbers are within file bounds
3. No broken references

Exit code 0 if all references valid, 1 if any broken.

Usage:
    python verify_code_refs.py
    python verify_code_refs.py --fix  # Update invalid line numbers (not implemented yet)
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Repository root (parent of .doc/)
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DOC_ROOT = REPO_ROOT / ".doc"

# Pattern to match code references
# Matches: `path/to/file.ext:123` or `path/to/file.ext:123-456`
CODE_REF_PATTERN = re.compile(
    r'`([a-zA-Z0-9_/.]+\.(cpp|py|h|hpp|c|cc|js|ts|json|yaml|yml|sh|md)):(\d+)(?:-(\d+))?`'
)

# Also match markdown link format: [text](path:line)
LINK_REF_PATTERN = re.compile(
    r'\[([^\]]+)\]\(([a-zA-Z0-9_/.]+\.(cpp|py|h|hpp|c|cc|js|ts|json|yaml|yml|sh|md))(?:#L(\d+)(?:-L(\d+))?)?\)'
)


class CodeRef:
    """Represents a code reference found in documentation"""

    def __init__(self, doc_file: Path, line_num: int, file_path: str, start_line: int, end_line: Optional[int] = None):
        self.doc_file = doc_file
        self.doc_line = line_num
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line or start_line

    def __str__(self):
        if self.start_line == self.end_line:
            ref = f"{self.file_path}:{self.start_line}"
        else:
            ref = f"{self.file_path}:{self.start_line}-{self.end_line}"
        return f"{self.doc_file}:{self.doc_line} → {ref}"


def find_code_refs(md_file: Path) -> List[CodeRef]:
    """Extract all code references from a markdown file"""
    refs = []

    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                # Match inline code references: `path/file.ext:123`
                for match in CODE_REF_PATTERN.finditer(line):
                    file_path = match.group(1)
                    start_line = int(match.group(3))
                    end_line = int(match.group(4)) if match.group(4) else None
                    refs.append(CodeRef(md_file, line_num, file_path, start_line, end_line))

                # Match markdown link references: [text](path#L123-L456)
                for match in LINK_REF_PATTERN.finditer(line):
                    file_path = match.group(2)
                    start_line = int(match.group(4)) if match.group(4) else None
                    end_line = int(match.group(5)) if match.group(5) else start_line

                    if start_line:  # Only add if line number specified
                        refs.append(CodeRef(md_file, line_num, file_path, start_line, end_line))

    except Exception as e:
        print(f"⚠️  Error reading {md_file}: {e}", file=sys.stderr)

    return refs


def verify_ref(ref: CodeRef) -> Tuple[bool, str]:
    """
    Verify a code reference is valid.

    Returns:
        (is_valid, error_message)
    """
    # Resolve file path relative to repo root
    target_file = REPO_ROOT / ref.file_path

    # Check file exists
    if not target_file.exists():
        return False, f"File not found: {ref.file_path}"

    # Check file is readable
    if not target_file.is_file():
        return False, f"Not a file: {ref.file_path}"

    # Count lines in file
    try:
        with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)
    except Exception as e:
        return False, f"Cannot read file {ref.file_path}: {e}"

    # Check line numbers are within bounds
    if ref.start_line > total_lines:
        return False, f"Line {ref.start_line} exceeds file length ({total_lines} lines)"

    if ref.end_line > total_lines:
        return False, f"Line {ref.end_line} exceeds file length ({total_lines} lines)"

    if ref.start_line > ref.end_line:
        return False, f"Invalid range: {ref.start_line}-{ref.end_line} (start > end)"

    return True, "OK"


def find_all_markdown_files() -> List[Path]:
    """Find all markdown files in .doc directory"""
    md_files = []

    for md_file in DOC_ROOT.rglob("*.md"):
        # Skip deprecated docs
        if ".doc_deprecated" in str(md_file):
            continue
        md_files.append(md_file)

    return sorted(md_files)


def main():
    """Main verification logic"""
    print("🔍 Code Reference Verifier")
    print(f"📁 Repository: {REPO_ROOT}")
    print(f"📄 Documentation: {DOC_ROOT}")
    print()

    # Find all markdown files
    md_files = find_all_markdown_files()
    print(f"Found {len(md_files)} markdown files")
    print()

    # Extract all code references
    all_refs = []
    for md_file in md_files:
        refs = find_code_refs(md_file)
        all_refs.extend(refs)

    print(f"Found {len(all_refs)} code references")
    print()

    # Verify each reference
    broken_refs = []
    valid_refs = []

    for ref in all_refs:
        is_valid, error_msg = verify_ref(ref)

        if is_valid:
            valid_refs.append(ref)
        else:
            broken_refs.append((ref, error_msg))
            print(f"❌ {ref}")
            print(f"   {error_msg}")
            print()

    # Summary
    print("=" * 70)
    print(f"✅ Valid references: {len(valid_refs)}")
    print(f"❌ Broken references: {len(broken_refs)}")
    print()

    if broken_refs:
        print("⚠️  FAILED: Documentation has broken code references")
        print()
        print("Broken references:")
        for ref, error_msg in broken_refs:
            print(f"  • {ref}")
            print(f"    {error_msg}")
        print()
        print("Please update documentation to fix broken references.")
        return 1
    else:
        print("✅ SUCCESS: All code references are valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
