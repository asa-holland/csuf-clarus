#!/usr/bin/env python3
"""
generate_review_form.py
Generates a blank reviewer assessment form from one or more Clarus statistics CSVs.
Reviewers fill in the blank columns and pass the completed file to analyze_reviews.py.

Usage:
    python generate_review_form.py report1.csv report2.csv --n 30
    python generate_review_form.py *.csv --n 50 --out my_review_form.csv
    python generate_review_form.py --dir ./reports --n 40 --seed 7

IMPORTANT — Cohen's Kappa requires control items (tool_flagged=0).
After generating the form, manually add rows where tool_flagged=0, text_a contains a
non-flagged segment, and leave reviewer_confirmed blank for the reviewer to fill in.
A ratio of roughly 1 control per 4 flagged items is sufficient.
"""

import csv
import io
import os
import sys
import glob
import random
import argparse
from pathlib import Path

SECTION_NAMES = {"DOCUMENT STATISTICS", "SEMANTIC EXTRACTION", "TERMINOLOGY", "CONTRADICTIONS"}

REVIEW_FIELDS = [
    'item_id',
    'source_file',
    'issue_category',    # contradiction | control
    'issue_type',        # semantic | modality | (blank for controls)
    'tool_confidence',   # tool confidence score, blank for controls
    'tool_flagged',      # 1 = flagged by tool, 0 = control (not flagged)
    'text_a',            # Statement A / primary text span
    'text_b',            # Statement B / blank for non-contradiction issues
    # ── reviewer fills these in ──────────────────────────────────────────────
    'reviewer_confirmed',   # 1 = genuine issue, 0 = false positive / not an issue
    'span_located',         # 1 = reviewer could find the text span, 0 = could not
    'clarity_rating',       # 1 (very unclear explanation) to 5 (very clear)
    'reviewer_notes',       # free text
]

SESSION_FIELDS = [
    'session_id',
    'reviewer_id',
    'source_file',
    'n_items_reviewed',
    'effort_reduced',   # 1 = tool reduced manual scanning effort, 0 = did not
    'session_notes',
]


def split_sections(content):
    sections = {}
    current = None
    buf = []
    for line in content.splitlines():
        if line.strip() in SECTION_NAMES:
            if current is not None:
                sections[current] = buf
            current = line.strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = buf
    return sections


def get_doc_filename(sections, fallback):
    for line in sections.get('DOCUMENT STATISTICS', []):
        stripped = line.strip()
        if stripped.startswith('Filename,'):
            return stripped.partition(',')[2].strip()
    return fallback


def extract_contradictions(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    sections = split_sections(content)
    doc_name = get_doc_filename(sections, Path(filepath).stem)
    items = []

    contra_lines = sections.get('CONTRADICTIONS', [])
    header_idx = next(
        (i for i, l in enumerate(contra_lines) if l.strip().startswith('#,Type,Confidence')),
        None
    )
    if header_idx is None:
        return items

    table_text = '\n'.join(contra_lines[header_idx:])
    for row in csv.DictReader(io.StringIO(table_text)):
        items.append({
            'source_file':    doc_name,
            'issue_category': 'contradiction',
            'issue_type':     row.get('Type', '').strip(),
            'tool_confidence': row.get('Confidence', '').strip(),
            'tool_flagged':   1,
            'text_a':         row.get('Statement A', '').strip(),
            'text_b':         row.get('Statement B', '').strip(),
            'reviewer_confirmed': '',
            'span_located':       '',
            'clarity_rating':     '',
            'reviewer_notes':     '',
        })
    return items


def main():
    parser = argparse.ArgumentParser(
        description='Generate a blank reviewer assessment form from Clarus statistics CSVs.'
    )
    parser.add_argument('files', nargs='*', help='Statistics CSV files or glob patterns')
    parser.add_argument('--dir',  help='Directory to scan for *statistics*.csv files')
    parser.add_argument('--n',    type=int, default=30,
                        help='Max items sampled per file (default: 30; 0 = all items)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--out',  default='review_form.csv',
                        help='Output form path (default: review_form.csv)')
    args = parser.parse_args()

    paths = []
    if args.dir:
        paths += glob.glob(os.path.join(args.dir, '*statistics*.csv'))
    for pattern in args.files:
        expanded = glob.glob(pattern)
        paths += expanded if expanded else [pattern]
    paths = sorted(set(paths))

    if not paths:
        print('No files found. Pass CSV files as arguments or use --dir.', file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    all_items = []

    for p in paths:
        if not os.path.isfile(p):
            print(f'  SKIP (not found): {p}', file=sys.stderr)
            continue
        try:
            items = extract_contradictions(p)
            if args.n and len(items) > args.n:
                items = rng.sample(items, args.n)
            all_items.extend(items)
            print(f'  {len(items):>4} items  <-  {p}', file=sys.stderr)
        except Exception as e:
            print(f'  ERROR: {p} — {e}', file=sys.stderr)

    if not all_items:
        print('No items extracted.', file=sys.stderr)
        sys.exit(1)

    rng.shuffle(all_items)
    for i, item in enumerate(all_items, 1):
        item['item_id'] = i

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_items)

    sessions_path = str(Path(args.out).with_suffix('')) + '_sessions.csv'
    with open(sessions_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        writer.writeheader()

    print(f'\n{len(all_items)} items written to:  {args.out}')
    print(f'Session template at:          {sessions_path}')
    print()
    print('Reviewer instructions:')
    print('  reviewer_confirmed  1 = genuine clarity issue   0 = false positive')
    print('  span_located        1 = found the text span     0 = could not locate it')
    print('  clarity_rating      1 = very unclear  ...  5 = very clear explanation')
    print('  reviewer_notes      optional free text')
    print()
    print('Session instructions (one row per review session):')
    print('  effort_reduced      1 = tool reduced manual scanning effort   0 = did not')
    print()
    print('NOTE: For Cohen\'s Kappa, manually add rows with tool_flagged=0')
    print('      (non-flagged control segments). Aim for ~1 control per 4 flagged items.')


if __name__ == '__main__':
    main()
