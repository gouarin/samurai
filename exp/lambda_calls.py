# usage: calls.py <binary> [dump-prefix]. Instruction count and out-of-line calls of the MUSCL flux lambdas (hllc, behind std::function).
import subprocess, sys, re, collections, os
E = os.path.expanduser('~/micromamba/envs/samurai-env/bin/')
out = subprocess.run([E + 'aarch64-conda-linux-gnu-objdump', '-d', '-C', '--no-show-raw-insn', sys.argv[1]], capture_output=True, text=True).stdout
funcs, cur = {}, None
for line in out.split('\n'):
    m = re.match(r'^[0-9a-f]+ <(.*)>:$', line)
    if m:
        cur = m.group(1); funcs[cur] = []; continue
    if cur is not None and re.match(r'^\s+[0-9a-f]+:\t', line):
        funcs[cur].append(line)
del out
roots = [f for f in funcs if 'make_muscl_scheme<(riemann::Solver)2' in f and ('_M_invoke' in f or 'operator()' in f) and 'static_for' not in f.split('make_muscl_scheme')[0]]
print(len(funcs), "functions;", len(roots), "roots")
for i, root in enumerate(roots):
    body = funcs[root]
    calls = collections.Counter()
    mn = collections.Counter()
    for l in body:
        p = l.split('\t')
        mn[p[1].strip()] += 1
        m = re.search(r'\tbl\t[0-9a-f]+ <(.*)>$', l)
        if m: calls[m.group(1)] += 1
        m = re.search(r'\tb(?:\.\w+)?\t[0-9a-f]+ <(.*)>$', l)
        if m and not re.sub(r'\+0x[0-9a-f]+$', '', m.group(1)) == root: calls['TAIL ' + m.group(1)] += 1
        if p[1].strip() in ('blr', 'br'): calls['indirect ' + p[1].strip()] += 1
    vec2 = sum(1 for l in body if re.search(r'\bv\d+\.2d\b', l))
    stk = sum(1 for l in body if re.search(r'\[(sp|x29)', l))
    print(f"\nroot {i}: {root[:230]}")
    print(f"  instructions: {len(body)}; fp/simd on .2d vectors: {vec2}; sp/x29 memory operands: {stk}; fsqrt: {mn['fsqrt']}; fdiv: {mn['fdiv']}; fmadd-family: {sum(v for k,v in mn.items() if k in ('fmadd','fmsub','fnmadd','fnmsub','fmla','fmls'))}; cond branches: {sum(v for k,v in mn.items() if k.startswith('b.') or k in ('cbz','cbnz','tbz','tbnz'))}; fcsel: {mn['fcsel']}; fcmp/fcmpe/fccmp: {mn['fcmp']+mn['fcmpe']+mn['fccmp']+mn['fccmpe']}")
    for t, c in calls.most_common():
        print(f"   {c:3d}  {t[:200]}")
    if len(sys.argv) > 2:
        open(f"{sys.argv[2]}-{i}.s", 'w').write(root + '\n' + '\n'.join(body) + '\n')
