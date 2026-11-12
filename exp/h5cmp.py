# usage: h5cmp.py a.h5 b.h5. Prints, per dataset, the shape and the maximum absolute difference.
import sys
import h5py
import numpy as np

def walk(g, out, prefix=""):
    for k, v in g.items():
        if isinstance(v, h5py.Dataset):
            out[prefix + k] = v[...]
        else:
            walk(v, out, prefix + k + "/")

a, b = {}, {}
walk(h5py.File(sys.argv[1]), a)
walk(h5py.File(sys.argv[2]), b)
for k in sorted(a):
    if k not in b:
        print(k, "missing")
    elif a[k].shape != b[k].shape:
        print(k, "shape", a[k].shape, b[k].shape)
    elif a[k].dtype.kind in "fiu":
        d = np.abs(a[k].astype(float) - b[k].astype(float)).max() if a[k].size else 0.0
        if d > 0:
            print(k, "max abs diff", d, "max abs value", np.abs(a[k]).max())
