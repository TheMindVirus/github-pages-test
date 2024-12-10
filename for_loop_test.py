import time

n = 2
y = 10
z = 0

t1 = 0
t2 = 0
t3 = 0

t1 = time.perf_counter_ns()
z = pow(n, y)
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "lookup table", z, "(", t3, "ns", ")")

z = 0
t1 = time.perf_counter_ns()
for a in range(0, n):
    for b in range(0, n):
        for c in range(0, n):
            for d in range(0, n):
                for e in range(0, n):
                    for f in range(0, n):
                        for g in range(0, n):
                            for h in range(0, n):
                                for i in range(0, n):
                                    for j in range(0, n):
                                        z += 1
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "precompiled routine", z, "(", t3, "ns", ")")

z = 0
x = n
t1 = time.perf_counter_ns()
for a in range(0, 100000000000):
    z += 1
    if z >= pow(n, y):
        break
    c = a % x
    d = int(a / x)
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "interpreted method", z, "(", t3, "ns", ")")

z = 0
c = [0] * y
l = [n] * y
found = False
t1 = time.perf_counter_ns()
for a in range(0, 100000000000):
    z += 1
    for b in range(0, y):
        c[b] += 1
        if c[b] >= l[b]:
            c[b] = 0
            if b >= y - 1:
                found = True
                break
        else:
            break
    if found:
        break
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "dimensional rotor", z, "(", t3, "ns", ")")

def calc_z():
    global z
    z += 1
    if z != pow(n, y):
        calc_z()

z = 0
t1 = time.perf_counter_ns()
calc_z()
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "recursive call", z, "(", t3, "ns", ")")

t1 = time.perf_counter_ns()
z = pow(n, y)
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "lookup table again", z, "(", t3, "ns", ")")

import numpy as np

t1 = time.perf_counter_ns()
z = np.power(n, y)
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "numpy lookup", z, "(", t3, "ns", ")")

t1 = time.perf_counter_ns()
z = np.power(n, y)
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "numpy lookup again", z, "(", t3, "ns", ")")

import pycuda.autoinit
import pycuda.compiler as cmp
import pycuda.driver as drv
import numpy as np
import os

cl_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
os.environ["PATH"] = os.environ["PATH"] + ";" + cl_path

t1 = time.perf_counter_ns()
z = np.power(n, y)
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "numpy after gpu load", z, "(", t3, "ns", ")")

kernel = \
"""
__global__ void power(float* dst, float* base, float* exp)
{
    const int i = threadIdx.x;
    dst[i] = __powf(base[i], exp[i]);
}
"""
mod = cmp.SourceModule(kernel)
power = mod.get_function("power")

base = np.array((n), dtype = np.float32)
exp = np.array((y), dtype = np.float32)
dst = np.zeros_like(base)

t1 = time.perf_counter_ns()

power(drv.Out(dst), drv.In(base), drv.In(exp),
      block = (1, 1, 1), grid = (1, 1))
z = dst
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "pycuda compute", z, "(", t3, "ns", ")")

import pyopencl as cl
import numpy as np
import sys
import os

os.environ["PYOPENCL_CTX"] = "0" # Device Select Default Graphics Card

tmp = sys.stdin
sys.stdin = None

cl.choose_devices(interactive = False)

kernel = \
"""
__kernel void power
(
    __global float* dst,
    __global const float* base,
    __global const float* exp
)
{
    int i = get_global_id(0);
    dst[i] = pow(base[i], exp[i]);
}
"""
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mod = cl.Program(ctx, kernel).build()

base = np.array([n], dtype = np.float32)
exp = np.array([y], dtype = np.float32)
dst = np.zeros_like(base)

gpu_in = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
gpu_out = cl.mem_flags.WRITE_ONLY

t1 = time.perf_counter_ns()

gpu_base = cl.Buffer(ctx, gpu_in, base.nbytes, hostbuf = base)
gpu_exp = cl.Buffer(ctx, gpu_in, exp.nbytes, hostbuf = exp)
gpu_dst = cl.Buffer(ctx, gpu_out, dst.nbytes)

mod.power(queue, dst.shape, None, gpu_dst, gpu_base, gpu_exp)

cl.enqueue_copy(queue, dst, gpu_dst)
z = dst[0]
t2 = time.perf_counter_ns()
t3 = t2 - t1
print("[INFO]:", "pyopencl kernel", z, "(", t3, "ns", ")")

sys.stdin = tmp
