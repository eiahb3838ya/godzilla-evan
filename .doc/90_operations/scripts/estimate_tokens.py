#!/usr/bin/env python3
"""
Estimate Token Counts for Documentation Files

Counts tokens in markdown files using tiktoken (OpenAI tokenizer).
Compares against estimates in index.yaml and reports discrepancies.

Usage:
    python estimate_tokens.py                    # Report only
    python estimate_tokens.py --update-index     # Update index.yaml with actual counts
    python estimate_tokens.py --file path.md     # Count single file

Requirements:
    pip install tiktoken pyyaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import yaml

try:
    import tiktoken
except ImportError:
    print("❌ Error: tiktoken not installed", file=sys.stderr)
    print("Install with: pip install tiktoken", file=sys.stderr)
    sys.exit(1)

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DOC_ROOT = REPO_ROOT / ".doc"
INDEX_YAML = DOC_ROOT / "00_index" / "index.yaml"

# Token encoding (GPT-4 compatible)
ENCODING = tiktoken.encoding_for_model("gpt-4")


def count_tokens(file_path: Path) -> int:
    """Count tokens in a markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tokens = ENCODING.encode(content)
        return len(tokens)

    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)
        return 0


def load_index_yaml() -> Dict:
    """Load index.yaml"""
    try:
        with open(INDEX_YAML, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Error loading {INDEX_YAML}: {e}", file=sys.stderr)
        return {}


def save_index_yaml(data: Dict):
    """Save updated index.yaml"""
    try:
        with open(INDEX_YAML, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✅ Updated {INDEX_YAML}")
    except Exception as e:
        print(f"❌ Error saving {INDEX_YAML}: {e}", file=sys.stderr)


def count_single_file(file_path: Path):
    """Count tokens for a single file and display"""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}", file=sys.stderr)
        return 1

    tokens = count_tokens(file_path)
    words = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            words = len(content.split())
    except:
        pass

    print(f"📄 {file_path}")
    print(f"   Tokens: {tokens:,}")
    print(f"   Words: {words:,}")
    print(f"   Ratio: {tokens / words:.2f} tokens/word" if words > 0 else "")

    return 0


def estimate_all_docs(update_index: bool = False) -> int:
    """Estimate tokens for all docs, optionally update index"""
    print("🔢 Token Estimator")
    print(f"📁 Documentation: {DOC_ROOT}")
    print()

    # Load index.yaml
    index_data = load_index_yaml()
    if not index_data or 'documents' not in index_data:
        print("⚠️  Warning: index.yaml not found or invalid", file=sys.stderr)
        index_data = {'documents': []}

    # Create lookup of documented files
    documented_files = {}
    for doc in index_data.get('documents', []):
        if 'path' in doc:
            documented_files[doc['path']] = doc

    # Scan all markdown files
    md_files = sorted(DOC_ROOT.rglob("*.md"))
    total_tokens = 0
    discrepancies = []
    updated_count = 0

    print(f"Found {len(md_files)} markdown files")
    print()

    for md_file in md_files:
        # Skip deprecated docs
        if ".doc_deprecated" in str(md_file) or "deprecated" in str(md_file).lower():
            continue

        # Get relative path from .doc/
        rel_path = md_file.relative_to(DOC_ROOT)
        actual_tokens = count_tokens(md_file)
        total_tokens += actual_tokens

        # Check if in index
        if str(rel_path) in documented_files:
            doc_entry = documented_files[str(rel_path)]
            estimated_tokens = doc_entry.get('tokens_estimate', 0)
            difference = actual_tokens - estimated_tokens

            # Calculate error percentage
            if estimated_tokens > 0:
                error_pct = abs(difference) / estimated_tokens * 100
            else:
                error_pct = 100 if actual_tokens > 0 else 0

            # Report discrepancies >10%
            if error_pct > 10:
                status = "⚠️ " if error_pct > 20 else "📊"
                print(f"{status} {rel_path}")
                print(f"   Estimated: {estimated_tokens:,} tokens")
                print(f"   Actual: {actual_tokens:,} tokens")
                print(f"   Difference: {difference:+,} ({error_pct:.1f}%)")
                print()
                discrepancies.append((str(rel_path), estimated_tokens, actual_tokens, error_pct))

                # Update if requested
                if update_index:
                    doc_entry['tokens_estimate'] = actual_tokens
                    updated_count += 1
            else:
                print(f"✅ {rel_path}: {actual_tokens:,} tokens (estimate: {estimated_tokens:,})")

        else:
            # Not in index
            print(f"📝 {rel_path}: {actual_tokens:,} tokens (not in index.yaml)")
            print()

    # Summary
    print("=" * 70)
    print(f"Total tokens across all docs: {total_tokens:,}")
    print(f"Files with discrepancies >10%: {len(discrepancies)}")
    print()

    if discrepancies:
        print("Largest discrepancies:")
        sorted_discrep = sorted(discrepancies, key=lambda x: x[3], reverse=True)[:5]
        for path, est, actual, error_pct in sorted_discrep:
            print(f"  • {path}: {error_pct:.1f}% error ({est:,} → {actual:,})")
        print()

    # Update index if requested
    if update_index and updated_count > 0:
        save_index_yaml(index_data)
        print(f"✅ Updated {updated_count} token estimates in index.yaml")
    elif update_index:
        print("ℹ️  No updates needed (all estimates within 10%)")

    # Exit code: 0 if all good, 1 if discrepancies
    return 1 if len(discrepancies) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Estimate tokens in documentation")
    parser.add_argument('--file', type=str, help="Count tokens in single file")
    parser.add_argument('--update-index', action='store_true', help="Update index.yaml with actual counts")
    args = parser.parse_args()

    if args.file:
        return count_single_file(Path(args.file))
    else:
        return estimate_all_docs(update_index=args.update_index)


if __name__ == "__main__":
    sys.exit(main())
