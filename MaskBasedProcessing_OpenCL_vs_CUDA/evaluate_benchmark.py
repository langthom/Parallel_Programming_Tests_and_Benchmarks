
import os
import matplotlib
import matplotlib.pyplot as plt

class Entry:
  def __init__(self, line):
    vals           = line.split(',')
    self.mem       = float(vals[0])
    self.env_K     = int(vals[1])
    self.cpu_time  = float(vals[2])
    self.gpu_times = [ (int(vals[3+2*i]), float(vals[3+2*i+1])) for i in range((len(vals)-3)//2) ]
  
  @property
  def memory(self):
    return self.mem
  
  @property
  def K(self):
    return self.env_K
  
  def compute_speedups(self):
    return [ self.cpu_time / gpu_time[1] for gpu_time in self.gpu_times ]
  
  @property
  def threads_per_block(self):
    return [ gpu_time[0] for gpu_time in self.gpu_times[:len(self.gpu_times)//2] ]
  

def evalute_benchmark(csv_file):
  entries = []
  with open(csv_file, "r") as f:
    for line_index, line in enumerate(f):
      if line_index % 2 == 0:
        entries.append( Entry(line) )
  
  out_dir = os.path.dirname(csv_file)
  
  # generate speedup plots as the number of threads per block increase
  entries_per_memory = {}
  for entry in entries:
    if entry.memory not in entries_per_memory:
      entries_per_memory[entry.memory] = [ (entry.K, entry.compute_speedups()) ]
    else:
      entries_per_memory[entry.memory].append((entry.K, entry.compute_speedups()))
  
  threads_per_block = entries[0].threads_per_block
  num_experiments = len(threads_per_block)
  max_speedup = max([ max(_speedups) for m in entries_per_memory for _K, _speedups in entries_per_memory[m] ])
  
  for memory in entries_per_memory:
    plt.figure()
    
    for K, speedups in entries_per_memory[memory]:
      plt.plot(threads_per_block, speedups[:num_experiments], label=f"CUDA     K={K}")
      plt.plot(threads_per_block, speedups[num_experiments:], label=f"OpenCL  K={K}")
    
    plt.title(f"Speedups for {memory} GiB")
    plt.xlabel("threads per block")
    plt.xticks(threads_per_block)
    plt.xscale('log')
    plt.ylim([0, max_speedup + 10])
    plt.ylabel("Speedup [x]")
    plt.legend()
    plt.savefig(f"{out_dir}/speedups_{memory}.pdf", dpi=300)
    plt.clf()



if __name__ == '__main__':
  import sys
  if len(sys.argv) < 2:
    print(f"Usage {sys.argv[0]} <path/to/evaluation/csv>")
    exit(1)
  evalute_benchmark(sys.argv[1])
