# usage: stack_forwarding.py <function.s> (a file written by lambda_calls.py --dump)
# Counts the 16-byte loads from the stack whose bytes were last written, in linear order, by 8-byte stores:
# the pattern that defeats store-to-load forwarding.
import re, sys

ins = []
for l in open(sys.argv[1]).read().split('\n')[1:]:
    p = l.split('\t')
    if len(p) > 1:
        ins.append((p[1].strip(), p[2] if len(p) > 2 else ''))
events = []
for i, (m, o) in enumerate(ins):
    mo = re.search(r'\[sp(?:, #(\d+))?\]', o)
    kind = 'S' if m in ('str', 'stp', 'stur') else 'L' if m in ('ldr', 'ldp', 'ldur') else None
    if not mo or not kind:
        continue
    pos = int(mo.group(1) or 0)
    for r in re.findall(r'\b([dqxw])\d+\b', o.split('[')[0]):
        sz = {'q': 16, 'd': 8, 'x': 8, 'w': 4}[r]
        events.append((i, kind, pos, sz)); pos += sz
wide = narrow = 0
dists = []
for k, (i, t, pos, sz) in enumerate(events):
    if t != 'L' or sz != 16:
        continue
    wide += 1
    cover = {}
    for (j, t2, p2, s2) in reversed(events[:k]):
        if t2 == 'S' and p2 < pos + 16 and p2 + s2 > pos:
            for b in range(max(p2, pos), min(p2 + s2, pos + 16), 8):
                cover.setdefault(b, (j, s2))
        if len(cover) == 2:
            break
    if any(s < 16 for _, s in cover.values()):
        narrow += 1
        dists.append(i - max(j for j, _ in cover.values()))
print(f"{sys.argv[1].split('/')[-1]}: 16-byte stack loads: {wide}; fed by 8-byte stores: {narrow}; distance to the last such store, in instructions: {sorted(dists)}")
