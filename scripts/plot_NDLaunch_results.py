
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def read_csv(fname):
  lines = []
  with open(fname, "r") as csv:
    for i, line in enumerate(csv):
      if i == 0:
        header = line.strip().split(',')
      else:
        lines.append( list(map(float, line.split(','))) )
  return header, np.array(lines)



def plot_ndlaunch(fname):
  header, data = read_csv(fname)
  sizes = np.unique(data[:,0])
  Ks    = np.unique(data[:,1])
  gpu_offset = 3
  
  throughputs = data[:,0,np.newaxis] * 1000.0 / data[:,gpu_offset:] # [GiB/s]
  
  w = 0.25
  xoff = -w
  plt.figure(figsize=(10,10))
  for dim in range(3):
    throughputs_for_dim = throughputs[:,dim].reshape(sizes.size, Ks.size) # n_sizes x n_Ks
    xs    = Ks + xoff + dim * w
    ys    = np.mean(throughputs_for_dim, axis=0) # n_Ks x 1
    yerrs = np.std( throughputs_for_dim, axis=0) # n_Ks x 1
    plt.errorbar(xs, ys, yerr=yerrs, capsize=2, fmt="o", markersize=2, label=f"{dim+1}D launch")

  plt.xticks(ticks=Ks, labels=Ks.astype(int))
  plt.xlabel("Environment size")
  plt.ylabel("Throughput [GiB/s]")
  plt.legend()
  plt.title("ND launch comparison\n(averaged over input sizes)")
  plt.savefig(f"{fname[:-4]}.png", dpi=300)
  


if __name__ == '__main__':
  import sys 
  plot_ndlaunch(sys.argv[1])
