import re

with open(r'g:\nckh\paper\main.tex', 'r', encoding='utf-8') as f:
    tex_content = f.read()

with open(r'g:\nckh\paper\refs.bib', 'r', encoding='utf-8') as f:
    bib_content = f.read()

cites = set()
for match in re.findall(r'\\cite\{([^}]+)\}', tex_content):
    for key in match.split(','):
        cites.add(key.strip())

bib_keys = set(re.findall(r'@[a-zA-Z]+\{([^,]+),', bib_content))

missing = cites - bib_keys
print("Missing citation keys in refs.bib:")
for m in missing:
    print("  -", m)
