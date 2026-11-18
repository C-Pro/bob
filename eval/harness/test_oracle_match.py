#!/usr/bin/env python3
import re

oracle_out = """
/workspace/testbed/dir3/subdir1/subsubdir1/file.txt /workspace/testbed/dir3/subdir1/subsubdir1/tmp/tmp.txt 
/workspace/testbed/dir1/AnotherHello.java /workspace/testbed/Hello1.java /workspace/testbed/Hello.java 
"""

bob_stdout = """
Duplicate files found:

[code]
./dir3/subdir1/subsubdir1/file.txt ./dir3/subdir1/subsubdir1/tmp/tmp.txt
./Hello.java ./Hello1.java ./dir1/AnotherHello.java
[/code]
"""

def extract_path_tokens(text: str) -> set:
    tokens = re.findall(r'[\w\./\-]+\.\w+', text)
    cleaned = set()
    for t in tokens:
        c = re.sub(r'^(?:/workspace)?/(?:testbed|system)/|^\./', '', t).strip('/')
        if len(c) > 3 and not c.startswith('http'):
            cleaned.add(c)
    return cleaned

o_paths = extract_path_tokens(oracle_out)
b_paths = extract_path_tokens(bob_stdout)
print('Oracle paths:', sorted(o_paths))
print('Bob paths:   ', sorted(b_paths))
print('Common:      ', sorted(o_paths & b_paths))
print('Match ratio: ', len(o_paths & b_paths) / len(o_paths))
