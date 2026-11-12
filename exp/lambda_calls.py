# usage: lambda_calls.py <binary> [--dump <prefix>] [--match <regex>]
# Static counts and out-of-line calls of the MUSCL flux lambdas (hllc, behind std::function).
# --match reports every function whose demangled name matches the regex instead.
# --dump writes the disassembly of each reported function to <prefix>-<i>.s.
import subprocess, sys, re, collections, os

E = os.path.expanduser('~/micromamba/envs/samurai-env/bin/')
args = sys.argv[1:]
binary = args.pop(0)
dump = match = None
while args:
    o = args.pop(0)
    if o == '--dump':
        dump = args.pop(0)
    elif o == '--match':
        match = re.compile(args.pop(0))

out = subprocess.run([E + 'aarch64-conda-linux-gnu-objdump', '-d', '-C', '--no-show-raw-insn', binary], capture_output=True, text=True).stdout
funcs, cur = {}, None
for line in out.split('\n'):
    m = re.match(r'^[0-9a-f]+ <(.*)>:$', line)
    if m:
        cur = m.group(1); funcs[cur] = []; continue
    if cur is not None and re.match(r'^\s+[0-9a-f]+:\t', line):
        funcs[cur].append(line)
del out

if match:
    roots = [f for f in funcs if match.search(f)]
else:
    # the lambda itself: GCC names it ...::operator(), Clang inlines it into _M_invoke
    roots = [f for f in funcs if 'make_muscl_scheme<(riemann::Solver)2' in f and ('_M_invoke' in f or 'operator()' in f) and len(funcs[f]) > 200]

SCALAR = r'\t(fadd|fsub|fmul|fdiv|fn?madd|fn?msub|fnmul|fabs|fneg|fsqrt|fmin\w*|fmax\w*)\td'
VECTOR = r'\t(fadd|fsub|fmul|fdiv|fmla|fmls|fabs|fneg|fsqrt|fmin\w*|fmax\w*|bsl|bit|bif|fcm\w+)\tv'
print(len(funcs), "functions;", len(roots), "reported")
for i, root in enumerate(roots):
    body = funcs[root]
    calls, mn = collections.Counter(), collections.Counter()
    for l in body:
        p = l.split('\t')
        mn[p[1].strip()] += 1
        m = re.search(r'\tbl\t[0-9a-f]+ <(.*)>$', l)
        if m: calls[m.group(1)] += 1
        m = re.search(r'\tb(?:\.\w+)?\t[0-9a-f]+ <(.*)>$', l)
        if m and re.sub(r'\+0x[0-9a-f]+$', '', m.group(1)) != root: calls['TAIL ' + m.group(1)] += 1
        if p[1].strip() in ('blr', 'br'): calls['indirect ' + p[1].strip()] += 1
    cnt = lambda pat: sum(1 for l in body if re.search(pat, l))
    print(f"\n[{i}] {root[:260]}")
    print(f"  instructions: {len(body)}; stack stores/loads: {cnt(r'\t(str|stp|stur)\t.*\[(sp|x29)')}/{cnt(r'\t(ldr|ldp|ldur)\t.*\[(sp|x29)')}"
          f"; scalar fp arith: {cnt(SCALAR)}; vector fp arith: {cnt(VECTOR)}; fmov+mov: {mn['fmov'] + mn['mov']}"
          f"; fsqrt: {mn['fsqrt']}; fdiv: {mn['fdiv']}; cond branches: {sum(v for k, v in mn.items() if k.startswith('b.') or k in ('cbz', 'cbnz', 'tbz', 'tbnz'))}; fcsel: {mn['fcsel']}")
    for t, c in calls.most_common():
        print(f"   {c:3d}  {t[:200]}")
    if dump:
        open(f"{dump}-{i}.s", 'w').write(root + '\n' + '\n'.join(body) + '\n')
