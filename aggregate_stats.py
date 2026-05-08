#!/usr/bin/env python3
"""
aggregate_stats.py
Aggregates multiple Clarus statistics CSV files into a single summary table.

Usage:
    python aggregate_stats.py reports/*.csv
    python aggregate_stats.py file1.csv file2.csv
    python aggregate_stats.py --dir path/to/folder
    python aggregate_stats.py *.csv --out summary.csv
"""

import csv
import io
import os
import sys
import glob
import argparse
from pathlib import Path

SECTION_NAMES = {"DOCUMENT STATISTICS", "SEMANTIC EXTRACTION", "TERMINOLOGY", "CONTRADICTIONS"}


def split_sections(content: str) -> dict:
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


def parse_kv(lines: list) -> dict:
    """Parse leading-whitespace-tolerant key,value lines."""
    result = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or ',' not in stripped:
            continue
        key, _, val = stripped.partition(',')
        result[key.strip()] = val.strip()
    return result


def to_int(s, default=0):
    try:
        return int(str(s).replace(',', '').strip())
    except (ValueError, AttributeError):
        return default


def to_float(s, default=0.0):
    try:
        return float(str(s).strip().rstrip('%'))
    except (ValueError, AttributeError):
        return default


def parse_file(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    m = {}
    sections = split_sections(content)

    # DOCUMENT STATISTICS
    kv = parse_kv(sections.get('DOCUMENT STATISTICS', []))
    m['filename']       = kv.get('Filename', Path(filepath).stem)
    m['text_length']    = to_int(kv.get('Text Length (chars)', 0))
    m['total_segs']     = to_int(kv.get('Total Segments', 0))
    m['normative']      = to_int(kv.get('Normative', 0))
    m['informative']    = to_int(kv.get('Informative', 0))
    m['unknown']        = to_int(kv.get('Unknown', 0))
    m['high_conf_segs'] = to_int(kv.get('High Confidence Segments (≥0.7)', 0))
    m['med_conf_segs']  = to_int(kv.get('Medium Confidence Segments', 0))
    m['low_conf_segs']  = to_int(kv.get('Low Confidence Segments (<0.4)', 0))

    # SEMANTIC EXTRACTION
    kv = parse_kv(sections.get('SEMANTIC EXTRACTION', []))
    m['sem_profiles']    = to_int(kv.get('Total Profiles', 0))
    m['avg_profile_conf'] = to_float(kv.get('Average Profile Confidence', 0))

    # TERMINOLOGY
    kv = parse_kv(sections.get('TERMINOLOGY', []))
    m['defined_terms']   = to_int(kv.get('Defined Terms', 0))
    m['undefined_terms'] = to_int(kv.get('Undefined Terms', 0))
    m['def_coverage']    = to_float(kv.get('Definition Coverage', 0))

    # CONTRADICTIONS — header kv + CSV table
    contra_lines = sections.get('CONTRADICTIONS', [])
    kv = parse_kv(contra_lines)
    m['total_contra'] = to_int(kv.get('Total Detected', 0))

    sem_n = mod_n = high_n = 0
    confs = []
    header_idx = next(
        (i for i, l in enumerate(contra_lines) if l.strip().startswith('#,Type,Confidence')),
        None
    )
    if header_idx is not None:
        table_text = '\n'.join(contra_lines[header_idx:])
        for row in csv.DictReader(io.StringIO(table_text)):
            t = row.get('Type', '').strip()
            if t == 'semantic':
                sem_n += 1
            elif t == 'modality':
                mod_n += 1
            c = to_float(row.get('Confidence', 0))
            if c > 0:
                confs.append(c)
                if c >= 95.0:
                    high_n += 1

    m['sem_contra']       = sem_n
    m['mod_contra']       = mod_n
    m['avg_contra_conf']  = round(sum(confs) / len(confs), 1) if confs else 0.0
    m['high_conf_contra'] = high_n

    # Derived
    total = m['total_segs']
    m['normative_pct'] = round(100 * m['normative'] / total, 1) if total else 0.0

    return m


# ── Column definitions ────────────────────────────────────────────────────────
# (field_key, display_header)
COLUMNS = [
    ('filename',         'Document'),
    ('text_length',      'Chars'),
    ('total_segs',       'Segments'),
    ('normative',        'Normative'),
    ('informative',      'Informative'),
    ('unknown',          'Unknown'),
    ('normative_pct',    'Norm%'),
    ('high_conf_segs',   'Hi-Conf Segs'),
    ('sem_profiles',     'Sem Profiles'),
    ('avg_profile_conf', 'Profile Conf%'),
    ('defined_terms',    'Defined'),
    ('undefined_terms',  'Undefined'),
    ('def_coverage',     'Def Cover%'),
    ('total_contra',     'Contradictions'),
    ('sem_contra',       'Contra:Sem'),
    ('mod_contra',       'Contra:Mod'),
    ('avg_contra_conf',  'Contra Conf%'),
    ('high_conf_contra', 'Contra>=95%'),
]


def build_table(rows: list) -> str:
    keys    = [c[0] for c in COLUMNS]
    headers = [c[1] for c in COLUMNS]

    str_rows = [[str(r.get(k, '')) for k in keys] for r in rows]

    widths = [len(h) for h in headers]
    for sr in str_rows:
        for i, cell in enumerate(sr):
            widths[i] = max(widths[i], len(cell))

    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def fmt(cells):
        return '|' + '|'.join(f' {c:<{widths[i]}} ' for i, c in enumerate(cells)) + '|'

    lines = [sep, fmt(headers), sep] + [fmt(sr) for sr in str_rows] + [sep]
    return '\n'.join(lines)


def write_csv(rows: list, outpath: str):
    keys    = [c[0] for c in COLUMNS]
    headers = [c[1] for c in COLUMNS]
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(k, '') for k in keys])


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate Clarus statistics CSVs into a summary table.'
    )
    parser.add_argument('files', nargs='*', help='CSV files or glob patterns')
    parser.add_argument('--dir', help='Directory to scan for *statistics*.csv files')
    parser.add_argument('--out', default='combined_stats.csv',
                        help='Output CSV filename (default: combined_stats.csv)')
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

    rows = []
    for p in paths:
        if not os.path.isfile(p):
            print(f'  SKIP (not found): {p}', file=sys.stderr)
            continue
        try:
            rows.append(parse_file(p))
            print(f'  parsed: {p}', file=sys.stderr)
        except Exception as e:
            print(f'  ERROR: {p} — {e}', file=sys.stderr)

    if not rows:
        print('No files parsed successfully.', file=sys.stderr)
        sys.exit(1)

    print(f'\n{len(rows)} file(s) aggregated\n')
    print(build_table(rows))
    print()

    write_csv(rows, args.out)
    print(f'CSV written to: {args.out}')


if __name__ == '__main__':
    main()
