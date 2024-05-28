
import numpy as np
import matplotlib.pyplot as plt

def parse_file(fname):
  # Mem [GiB],K,CPU [ms],GPU global memory 1D [ms],GPU global memory 3D [ms],GPU shared memory [ms]
  table = []
  with open(fname, "r") as csv:
    for i, line in enumerate(csv):
      l = line.split(',')
      if i == 0:
        header = list(map(lambda x: x.strip(), l))
      else:
        table.append( list( map(float, l) ) )
  return header, np.atleast_2d(table)

def plot_smem_comparison(fname, device_name):
  header, data = parse_file(fname)
  
  w = 0.1
  Ks = np.unique(data[:,1].astype(int))
  Xs = np.arange(Ks.size) - 1*w
  
  for method in range(3):
    Y_avg = [ data[Ki::Ks.size,3+method].mean() for Ki in range(Ks.size) ]
    Y_std = [ data[Ki::Ks.size,3+method].std()  for Ki in range(Ks.size) ]
    plt.errorbar(Xs + method*w, Y_avg, yerr=Y_std, capsize=6, fmt="o", label=header[3+method])
  
  plt.xticks(np.arange(Ks.size), Ks)
  plt.xlabel("Environment size [voxels]")
  plt.ylabel("Runtime [ms]")
  plt.title(f"Shared memory runtimes\n(device: {device_name})")
  plt.legend()
  plt.savefig(f"smem_comparison_runtime_{device_name.replace(' ', '_')}_{data[:,0].astype(int).max()}GiB.pdf", dpi=600)
  plt.close()

def plot_smem_comparison_throughput(fname, device_name):
  header, data = parse_file(fname)
  
  w = 0.125
  Ks = np.unique(data[:,1].astype(int))
  Xs = np.arange(Ks.size) - 1.5*w
  
  def _throughput(Ki, method):
    return data[Ki::Ks.size,0] * 1000.0 / data[Ki::Ks.size,3+method]
  
  colors = ["red", "blue", "black"]
  
  for method in range(3):
    Y_avg = [ _throughput(Ki, method).mean() for Ki in range(Ks.size) ]
    Y_std = [ _throughput(Ki, method).std()  for Ki in range(Ks.size) ]
    plt.errorbar(Xs + method*w, Y_avg, yerr=Y_std, capsize=6, fmt=".", label=header[3+method], color=colors[method])
    plt.plot(Xs+method*w, Y_avg, color=colors[method])
  
  plt.xticks(np.arange(Ks.size), Ks)
  plt.xlabel("Environment size [voxels]")
  plt.ylabel("Throughput [GiB/s]")
  plt.title(f"Shared memory throughputs\n(device: {device_name})")
  plt.legend()
  plt.savefig(f"smem_comparison_throughput_{device_name.replace(' ', '_')}_{data[:,0].astype(int).max()}GiB.pdf", dpi=600)
  plt.close()




if __name__ == '__main__':
  import sys
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <comparison.csv> <device>")
    exit(1)
    
  plot_smem_comparison(sys.argv[1], sys.argv[2])
  plot_smem_comparison_throughput(sys.argv[1], sys.argv[2])
