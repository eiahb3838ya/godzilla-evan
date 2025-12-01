#!/usr/bin/env python3
"""
Check Markdown Links

Validates all markdown links in documentation:
1. Internal links to other markdown files
2. Internal links to directories
3. Anchor links within documents (#heading)

Usage:
    python check_links.py
    python check_links.py --verbose  # Show all links checked

Exit code 0 if all links valid, 1 if any broken.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import unquote

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DOC_ROOT = REPO_ROOT / ".doc"

# Pattern to match markdown links: [text](path) or [text](path#anchor)
LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

# Pattern to extract anchors from markdown headings
HEADING_PATTERN = re.compile(r'^#+\s+(.+)$', re.MULTILINE)


class Link:
    """Represents a markdown link"""

    def __init__(self, source_file: Path, line_num: int, text: str, target: str):
        self.source_file = source_file
        self.line_num = line_num
        self.text = text
        self.target = target

        # Parse target into path and anchor
        if '#' in target:
            self.path, self.anchor = target.split('#', 1)
        else:
            self.path = target
            self.anchor = None

    def __str__(self):
        return f"{self.source_file}:{self.line_num} → [{self.text}]({self.target})"

    def is_external(self) -> bool:
        """Check if link is external (http/https)"""
        return self.target.startswith('http://') or self.target.startswith('https://')

    def is_anchor_only(self) -> bool:
        """Check if link is anchor-only (#heading in same file)"""
        return self.target.startswith('#')


def slugify(text: str) -> str:
    """Convert heading text to anchor slug (GitHub-style)"""
    # Lowercase
    slug = text.lower()

    # Remove special characters except spaces and hyphens
    slug = re.sub(r'[^\w\s-]', '', slug)

    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)

    # Remove multiple consecutive hyphens
    slug = re.sub(r'-+', '-', slug)

    # Strip leading/trailing hyphens
    slug = slug.strip('-')

    return slug


def extract_headings(md_file: Path) -> List[str]:
    """Extract all heading anchors from a markdown file"""
    anchors = []

    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        for match in HEADING_PATTERN.finditer(content):
            heading_text = match.group(1)
            anchor = slugify(heading_text)
            anchors.append(anchor)

    except Exception as e:
        print(f"⚠️  Error reading {md_file}: {e}", file=sys.stderr)

    return anchors


def find_links(md_file: Path) -> List[Link]:
    """Extract all links from a markdown file"""
    links = []

    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                for match in LINK_PATTERN.finditer(line):
                    text = match.group(1)
                    target = match.group(2)

                    # Decode URL encoding
                    target = unquote(target)

                    links.append(Link(md_file, line_num, text, target))

    except Exception as e:
        print(f"⚠️  Error reading {md_file}: {e}", file=sys.stderr)

    return links


def verify_link(link: Link, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify a markdown link is valid.

    Returns:
        (is_valid, error_message)
    """
    # Skip external links (we don't validate HTTP)
    if link.is_external():
        if verbose:
            print(f"ℹ️  Skipping external link: {link}")
        return True, "External link (not validated)"

    # Handle anchor-only links (#heading in same file)
    if link.is_anchor_only():
        if not link.anchor:
            return False, "Empty anchor link"

        # Extract headings from source file
        headings = extract_headings(link.source_file)
        if link.anchor not in headings:
            return False, f"Anchor '#{link.anchor}' not found in {link.source_file.name}"

        return True, "OK"

    # Resolve relative path
    if link.path:
        # Path is relative to the source file's directory
        source_dir = link.source_file.parent
        target_path = (source_dir / link.path).resolve()

        # Check if path exists
        if not target_path.exists():
            return False, f"File or directory not found: {link.path}"

        # If it's a file, check it's readable
        if target_path.is_file():
            # Check anchor if specified
            if link.anchor:
                headings = extract_headings(target_path)
                if link.anchor not in headings:
                    return False, f"Anchor '#{link.anchor}' not found in {target_path.name}"

            return True, "OK"

        # If it's a directory, that's OK too
        if target_path.is_dir():
            if link.anchor:
                return False, "Cannot have anchor link to directory"
            return True, "OK (directory)"

        return False, f"Unknown path type: {target_path}"

    # Empty path
    if not link.path and link.anchor:
        # This is handled by is_anchor_only()
        return True, "OK"

    return False, "Invalid link format"


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
    import argparse
    parser = argparse.ArgumentParser(description="Check markdown links")
    parser.add_argument('--verbose', action='store_true', help="Show all links checked")
    args = parser.parse_args()

    print("🔗 Markdown Link Checker")
    print(f"📁 Repository: {REPO_ROOT}")
    print(f"📄 Documentation: {DOC_ROOT}")
    print()

    # Find all markdown files
    md_files = find_all_markdown_files()
    print(f"Found {len(md_files)} markdown files")
    print()

    # Extract all links
    all_links = []
    for md_file in md_files:
        links = find_links(md_file)
        all_links.extend(links)

    print(f"Found {len(all_links)} links")
    print()

    # Verify each link
    broken_links = []
    valid_links = []
    external_links = 0

    for link in all_links:
        if link.is_external():
            external_links += 1
            continue

        is_valid, error_msg = verify_link(link, verbose=args.verbose)

        if is_valid:
            valid_links.append(link)
            if args.verbose:
                print(f"✅ {link}")
        else:
            broken_links.append((link, error_msg))
            print(f"❌ {link}")
            print(f"   {error_msg}")
            print()

    # Summary
    print("=" * 70)
    print(f"✅ Valid links: {len(valid_links)}")
    print(f"ℹ️  External links (not validated): {external_links}")
    print(f"❌ Broken links: {len(broken_links)}")
    print()

    if broken_links:
        print("⚠️  FAILED: Documentation has broken links")
        print()
        print("Broken links:")
        for link, error_msg in broken_links:
            print(f"  • {link}")
            print(f"    {error_msg}")
        print()
        print("Please fix broken links in documentation.")
        return 1
    else:
        print("✅ SUCCESS: All internal links are valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
