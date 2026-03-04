#!/usr/bin/env python3
"""
debug_comments.py
Verify per-post comment counts against the declared '共 N 条评论' numbers.

Usage:
    python3 debug_comments.py <txt_file>
    python3 debug_comments.py 冠军赛1.txt
"""

import sys
import os
import re

# Reuse parser internals
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xiaohongshu_parser import (
    split_into_blocks,
    parse_header,
    parse_comments,
    _COMMENT_COUNT_PAT,
)


def debug_file(file_path: str):
    try:
        with open(file_path, encoding='utf-8') as fh:
            content = fh.read()
    except UnicodeDecodeError:
        with open(file_path, encoding='utf-8-sig') as fh:
            content = fh.read()

    blocks = split_into_blocks(content)
    source = os.path.basename(file_path)
    print(f'\n=== {source} — {len(blocks)} post blocks ===\n')

    total_ok = 0
    total_fail = 0

    for idx, block in enumerate(blocks, start=1):
        lines = block.splitlines()

        # Find anchor
        anchor_idx = None
        declared = 0
        for i, line in enumerate(lines):
            m = _COMMENT_COUNT_PAT.search(line)
            if m:
                anchor_idx = i
                declared = int(m.group(1))
                break

        if anchor_idx is None:
            print(f'Post_{idx:02d}  [no anchor found]')
            continue

        header_lines = lines[:anchor_idx]
        comment_lines = lines[anchor_idx + 1:]
        post_info = parse_header(header_lines)
        comments = parse_comments(comment_lines)
        parsed = len(comments)

        match = '✓' if parsed == declared else '✗ MISMATCH'
        print(
            f'Post_{idx:02d}  author={post_info["author"]!r}  '
            f'declared={declared:3d}  parsed={parsed:3d}  {match}'
        )

        if parsed != declared:
            total_fail += 1
            # Show individual comment details
            for c_idx, c in enumerate(comments, start=1):
                short_content = c['content'][:60].replace('\n', '↵') if c['content'] else '(空)'
                print(
                    f'         {c_idx:3d}. [{c["content_type"]}] '
                    f'{c["username"]!r}  date={c["date"]!r}  '
                    f'likes={c["likes"]}  "{short_content}"'
                )
        else:
            total_ok += 1

    print(f'\nResult: {total_ok} OK / {total_fail} FAILED / {total_ok + total_fail} total\n')


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 debug_comments.py <txt_file>', file=sys.stderr)
        sys.exit(1)
    debug_file(sys.argv[1])


if __name__ == '__main__':
    main()
