
import os
import subprocess

# Metrics we deem useful for our purposes.
# Complete list of metrics: https://docs.nvidia.com/cuda/profiler-users-guide/ 
metrics = [
  "flop_count_sp",             # [    general    ] Total number of (single precision) floating point operations.
  "sm_efficiency",             # [bigger = better] Indication of how balanced the execution was. 
  "achieved_occupancy",        # [bigger = better] How many times at least 1 warp was active per SM (relative to the maximum number of active ones).
  "branch_efficiency",         # [bigger = better] How many threads were non-divergent?
  "warp_execution_efficiency", # [bigger = better] How many threads within a warp were active (relative to the maximum number of threads per warp)?
  "shared_load_throughput",    # [bigger = better] Throughput of shared memory read operations.
  "shared_store_throughput",   # [bigger = better] Throughput of shared memory store operations.
  "shared_efficiency",         # [bigger = better] The ratio of requested shared memory throughput to required shared memory throughput.
  "shared_utilization",        # [bigger = better] Utilization level of shared memory (compared to peak utilization, scale from 0 to 10).
  "gld_efficiency",            # [bigger = better] Rate of coalesced accesses to global memory when loading.
  "gst_efficiency",            # [bigger = better] Rate of coalesced accesses to global memory when storing.
  "local_load_throughput",     # [lower  = better] Load throughput local memory (which resides in global memory); low = more in registers
  "local_store_throughput",    # [lower  = better] Load throughput local memory (which resides in global memory); low = more in registers
  "shared_replay_overhead",    # [lower  = better] The average number of instruction replays (i.e., re-issues of instructions) for each instruction due to bank conflicts.
  "stall_exec_dependency",     # [lower  = better] How often was the input to a calculation not ready.
  "stall_memory_dependency",   # [lower  = better] How often was the memory not directly available or already fully utilized or too many requests to memory.
  "stall_sync",                # [lower  = better] How often was a stall issue due to __syncthreads.
  "stall_other",               # [lower  = better] Any other stall. 
  "stall_pipe_busy"            # [lower  = better] How often was the compute pipeline busy and a computation could not be performed.
]

def generate_nvprof_profile(kernel_name, exe_name, exe, args):
  cmd = [ "nvprof" ]
  if kernel_name != "all":
    cmd += ["--kernels", kernel_name]
  cmd += ["--metrics", ','.join(metrics)]
  cmd += ["--log-file", f"nvprof_{exe_name}_{kernel_name}.csv"]
  #cmd.append("--csv")
  cmd.append(exe)
  cmd += args
  
  subprocess.run(cmd)


if __name__ == '__main__':
  import sys 
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <kernel:all|kernel-name> <exe> [<exe-args>...]")
    exit(1)
  
  exe_name = os.path.splitext(os.path.basename(sys.argv[2]))[0]
  
  generate_nvprof_profile(sys.argv[1], exe_name, sys.argv[2], sys.argv[3:])
