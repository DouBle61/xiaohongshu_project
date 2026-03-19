#!/usr/bin/env python3
"""
xiaohongshu_parser.py
Parse manually-copied Xiaohongshu text files into structured CSV/Excel data.

Usage:
    python3 xiaohongshu_parser.py                          # reads ./raw_texts/*.txt
    python3 xiaohongshu_parser.py raw_texts/冠军赛1.txt   # single file
"""

import os
import re
import sys
import csv

try:
    from openpyxl import Workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches date lines, e.g.:
#   2025-05-13          2025-05-13湖南      02-05陕西
#   02-05 江西          编辑于 2025-04-27
_DATE_PAT = re.compile(
    r'^(?:编辑于\s*)?'
    r'(?P<date>\d{4}-\d{2}-\d{2}|\d{2}-\d{2})'
    r'\s*(?P<location>.*)?$'
)

# Matches lines that are purely a number (likes or reply count)
_NUM_PAT = re.compile(r'^\d+$')

# Matches "共 N 条评论"
_COMMENT_COUNT_PAT = re.compile(r'共\s*(\d+)\s*条评论')

# Matches image-ratio noise like "1/2", "3/4"
_RATIO_PAT = re.compile(r'^\d+/\d+$')

# Navigation bar items and other single-line noise to skip inside header
_NAV_ITEMS = {'发现', '直播', '发布', '通知', '我'}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _is_date_line(line: str) -> bool:
    return bool(_DATE_PAT.match(line.strip()))


def _parse_date_location(line: str):
    """Return (date_str, location_str) from a date line."""
    m = _DATE_PAT.match(line.strip())
    if m:
        return m.group('date'), (m.group('location') or '').strip()
    return '', ''


def _extract_hashtags(text: str):
    """Return list of #tags found in text."""
    return re.findall(r'#\S+', text)


def _strip_hashtags(text: str) -> str:
    """Remove #tags from text and return cleaned string."""
    return re.sub(r'#\S+', '', text).strip()


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

def split_into_blocks(content: str):
    """
    Split file content by '- THE END -' and return only blocks that contain
    a '共 N 条评论' marker (valid post blocks).

    Bug 1 fix: filter by 共 N 条评论 presence, not by character length.
    """
    raw_blocks = content.split('- THE END -')
    valid = []
    for block in raw_blocks:
        if _COMMENT_COUNT_PAT.search(block):
            valid.append(block)
    return valid


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_header(lines):
    """
    Parse the header section (lines above '共 N 条评论') to extract:
    author, title, body, hashtags, date, location.

    Strategy:
    1. Find the last occurrence of '我' (end of bottom nav bar).
    2. The next non-empty line is the author.
    3. Skip '关注'.
    4. Skip noise (empty lines, LIVE, image ratios).
    5. Collect remaining lines as body until a date line is found.
    6. Extract hashtags from body lines; first clean line = title.
    """
    # Find the last nav-bar '我'
    last_nav_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == '我':
            last_nav_idx = idx

    start = (last_nav_idx + 1) if last_nav_idx is not None else 0

    i = start
    n = len(lines)

    # Skip empty lines
    while i < n and not lines[i].strip():
        i += 1

    if i >= n:
        return {'author': '', 'title': '', 'body': '', 'hashtags': [],
                'date': '', 'location': ''}

    author = lines[i].strip()
    i += 1

    # Skip '关注'
    if i < n and lines[i].strip() == '关注':
        i += 1

    # Skip further noise: empty lines, LIVE, image ratios like "1/2"
    while i < n:
        s = lines[i].strip()
        if not s or s == 'LIVE' or _RATIO_PAT.match(s):
            i += 1
        else:
            break

    # Collect body lines until we hit the date line
    body_lines = []
    date_str = ''
    location = ''

    while i < n:
        s = lines[i].strip()
        if s and _is_date_line(s):
            date_str, location = _parse_date_location(s)
            i += 1
            break
        elif s:
            body_lines.append(s)
        i += 1

    # Extract hashtags from all body lines
    all_tags = []
    clean_lines = []
    for bl in body_lines:
        tags = _extract_hashtags(bl)
        all_tags.extend(tags)
        clean = _strip_hashtags(bl)
        if clean:
            clean_lines.append(clean)

    title = clean_lines[0] if clean_lines else ''
    body = '\n'.join(clean_lines[1:]) if len(clean_lines) > 1 else ''

    return {
        'author': author,
        'title': title,
        'body': body,
        'hashtags': all_tags,
        'date': date_str,
        'location': location,
    }


# ---------------------------------------------------------------------------
# Comment parsing
# ---------------------------------------------------------------------------

def parse_comments(lines):
    """
    Parse the comment section lines into a list of comment dicts.

    Each comment dict has:
        username, is_post_author, content, date, location, likes,
        reply_to, content_type ('Comment' or 'Reply')

    Comment structure:
        <username>
        [作者]          (optional)
        [empty]
        [content]       (may be empty; may start with '回复 XXX : ')
        [置顶评论]      (optional noise)
        <date+location>
        [empty]
        <赞 | number>   (likes; '赞' = 0)
        [empty]
        [number]        (optional reply count – skip; Bug 2 fix: max 2 numbers)
        [empty]
        [回复]          (optional UI button – skip)

    Bug 2 fix: consume at most 2 numbers (likes + reply count) in the tail.
    Bug 3 fix: allow empty content (stored as '').
    """
    comments = []
    i = 0
    n = len(lines)

    while i < n:
        # Skip blank lines between comments
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break

        # --- Username ---
        username = lines[i].strip()
        i += 1

        # --- Optional '作者' marker ---
        is_post_author = False
        if i < n and lines[i].strip() == '作者':
            is_post_author = True
            i += 1

        # Skip blank lines before content
        while i < n and not lines[i].strip():
            i += 1

        # --- Collect content lines until date+location line ---
        content_parts = []
        date_str = ''
        location = ''

        while i < n:
            s = lines[i].strip()
            if s and _is_date_line(s):
                date_str, location = _parse_date_location(s)
                i += 1
                break
            elif s == '置顶评论':
                i += 1  # skip noise
            elif s:
                content_parts.append(s)
                i += 1
            else:
                i += 1  # skip blank lines within content

        # --- Consume tail: likes + optional reply count ---
        # Skip blank lines
        while i < n and not lines[i].strip():
            i += 1

        likes = 0
        if i < n:
            s = lines[i].strip()
            if s == '赞':
                likes = 0
                i += 1
            elif _NUM_PAT.match(s):
                likes = int(s)
                i += 1

        # Skip blank lines
        while i < n and not lines[i].strip():
            i += 1

        # Optionally consume one more number (reply count) — Bug 2 fix
        if i < n and _NUM_PAT.match(lines[i].strip()):
            i += 1  # discard reply count
            # Skip blank lines
            while i < n and not lines[i].strip():
                i += 1

        # Optionally consume '回复' UI button
        if i < n and lines[i].strip() == '回复':
            i += 1

        # --- Build comment record ---
        content = '\n'.join(content_parts)  # may be empty string (Bug 3 fix)

        # Detect Reply vs Comment
        reply_to = ''
        content_type = 'Comment'
        reply_match = re.match(r'^回复\s+(.+?)\s*:\s*(.*)$', content, re.DOTALL)
        if reply_match:
            reply_to = reply_match.group(1).strip()
            content = reply_match.group(2).strip()
            content_type = 'Reply'

        comments.append({
            'username': username,
            'is_post_author': is_post_author,
            'content': content,
            'date': date_str,
            'location': location,
            'likes': likes,
            'reply_to': reply_to,
            'content_type': content_type,
        })

    return comments


# ---------------------------------------------------------------------------
# Block → rows
# ---------------------------------------------------------------------------

def parse_block(block: str, post_index: int, source_file: str):
    """
    Parse one post block into a list of row dicts.

    post_index is 1-based.
    Returns list of dicts with keys matching the output column names.
    """
    lines = block.splitlines()

    # Find '共 N 条评论' line
    anchor_idx = None
    declared_count = 0
    for idx, line in enumerate(lines):
        m = _COMMENT_COUNT_PAT.search(line)
        if m:
            anchor_idx = idx
            declared_count = int(m.group(1))
            break

    if anchor_idx is None:
        return None, 0, 0

    header_lines = lines[:anchor_idx]
    comment_lines = lines[anchor_idx + 1:]

    # Parse header
    post_info = parse_header(header_lines)

    # Build post row
    post_id = f'Post_{post_index:02d}'
    post_row = {
        'Text_ID': post_id,
        'Content_Type': 'Post',
        'Text': (post_info['title'] + ('\n' + post_info['body'] if post_info['body'] else '')).strip(),
        'Author': post_info['author'],
        'Date': post_info['date'],
        'Location': post_info['location'],
        'Likes': '',
        'Hashtags': '|'.join(post_info['hashtags']),
        'Reply_To': '',
        'Is_Post_Author': '',
        'Source_File': source_file,
    }

    rows = [post_row]

    # Parse comments
    comments = parse_comments(comment_lines)

    for c_idx, comment in enumerate(comments, start=1):
        comment_id = f'Comment_{post_index:02d}_{c_idx:03d}'
        rows.append({
            'Text_ID': comment_id,
            'Content_Type': comment['content_type'],
            'Text': comment['content'],
            'Author': comment['username'],
            'Date': comment['date'],
            'Location': comment['location'],
            'Likes': comment['likes'],
            'Hashtags': '',
            'Reply_To': comment['reply_to'],
            'Is_Post_Author': 'Yes' if comment['is_post_author'] else '',
            'Source_File': source_file,
        })

    return rows, declared_count, len(comments)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

COLUMNS = [
    'Text_ID', 'Content_Type', 'Text', 'Author', 'Date', 'Location',
    'Likes', 'Hashtags', 'Reply_To', 'Is_Post_Author', 'Source_File',
]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(rows, path):
    if not HAS_OPENPYXL:
        print('openpyxl not installed – skipping Excel output', file=sys.stderr)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb = Workbook()

    post_rows = [r for r in rows if r['Content_Type'] == 'Post']
    comment_rows = [r for r in rows if r['Content_Type'] in ('Comment', 'Reply')]

    for sheet_name, sheet_rows in [
        ('全部数据', rows),
        ('帖子正文-主题建模', post_rows),
        ('评论数据-情感分析', comment_rows),
    ]:
        ws = wb.active if sheet_name == '全部数据' else wb.create_sheet(sheet_name)
        ws.title = sheet_name
        ws.append(COLUMNS)
        for row in sheet_rows:
            ws.append([row.get(c, '') for c in COLUMNS])

    wb.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_files(file_paths):
    all_rows = []
    post_counter = 0

    for file_path in file_paths:
        source_file = os.path.basename(file_path)
        try:
            with open(file_path, encoding='utf-8') as fh:
                content = fh.read()
        except UnicodeDecodeError:
            with open(file_path, encoding='utf-8-sig') as fh:
                content = fh.read()

        blocks = split_into_blocks(content)
        print(f'{source_file}: found {len(blocks)} post blocks')

        for block in blocks:
            post_counter += 1
            rows, declared, parsed = parse_block(block, post_counter, source_file)
            if rows is None:
                continue
            all_rows.extend(rows)
            post_row = rows[0]
            print(
                f'  Post_{post_counter:02d} | {post_row["Author"]!r} | '
                f'declared={declared} parsed={parsed}'
                + (f' *** MISMATCH ***' if declared != parsed else '')
            )

    return all_rows


def main():
    if len(sys.argv) > 1:
        file_paths = [sys.argv[1]]
    else:
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_texts')
        if not os.path.isdir(raw_dir):
            print(f'Directory not found: {raw_dir}', file=sys.stderr)
            sys.exit(1)
        file_paths = sorted(
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir)
            if f.endswith('.txt')
        )
        if not file_paths:
            print(f'No .txt files found in {raw_dir}', file=sys.stderr)
            sys.exit(1)

    all_rows = process_files(file_paths)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    csv_path = os.path.join(out_dir, 'xiaohongshu_data.csv')
    xlsx_path = os.path.join(out_dir, 'xiaohongshu_data.xlsx')

    write_csv(all_rows, csv_path)
    print(f'\nCSV written: {csv_path}  ({len(all_rows)} rows)')

    write_excel(all_rows, xlsx_path)
    if HAS_OPENPYXL:
        print(f'Excel written: {xlsx_path}')


if __name__ == '__main__':
    main()
