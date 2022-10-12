import os
import pyopencl as cl
import numpy as np

os.environ["PYOPENCL_CTX"] = "0"

class GPUMemory:
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    def __init__(self, mem):
        self.mem = mem
        mf = cl.mem_flags
        if mem is not None:
            self.memGPU = cl.Buffer(GPUMemory.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mem)
        else:
            self.memGPU = None

    def copyToCPU(self):
        cl.enqueue_copy(GPUMemory.queue, self.mem, self.memGPU)
    
    def copyToGPU(self):
        cl.enqueue_copy(GPUMemory.queue, self.memGPU, self.mem)

    def reset(self):
        self.mem = np.zeros(self.mem.shape, np.float32)
        self.copyToGPU()