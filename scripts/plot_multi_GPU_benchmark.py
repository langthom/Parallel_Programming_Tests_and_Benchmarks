
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_multi_GPU(gpu_name, pc_name, benchmark_csv, has_single_threaded):
  data_all          = np.genfromtxt(benchmark_csv, delimiter=',')
  Ks                = np.unique(data_all[:,1]).astype(int)
  sizes             = np.unique(data_all[:,0])
  off               = 4 + 2*int(has_single_threaded) # offset to gpu measurements, "size,K,cpu_seg,K,cpu_par,t1,gpu1,..." or "size,K,cpu_par,t1,gpu1,..."
  threads_per_block = data_all[0,off-1::3].astype(int)
  speedups_all      = data_all[:,off::3] / data_all[:,off+1::3]
  
  # Plot 1:
  #   Per environment size (K), plot the speedup of multi GPU over single GPU over the number of threads per block.
  #   Speedups are averaged over the different sizes.
  plt.figure(figsize=(12,6))
  for K in Ks:
    speedups_for_K = speedups_all[data_all[:,1]==K,:]
    avg_speedup_for_K = speedups_for_K.mean(axis=0)
    std_speedup_for_K = speedups_for_K.std(axis=0)
    plt.plot(threads_per_block, avg_speedup_for_K, label=f"K = {K:2d}")
    plt.fill_between(threads_per_block, np.maximum(0,avg_speedup_for_K-std_speedup_for_K), avg_speedup_for_K+std_speedup_for_K, alpha=0.3)
    
  plt.title(f"Multi GPU speedups over single GPU\nDevice: {gpu_name}\n(averaged over sizes)")
  plt.xticks(threads_per_block)
  plt.xlabel("threads per block")
  plt.ylabel("speedup [x]")
  plt.legend()
  plt.savefig(f"{pc_name}_MultiGPU__Speedups_over_Ks.svg", dpi=600, bbox_inches="tight")
  plt.close()
  
  # =================================================================================================================== #
  
  # Plot 2:
  #   Per dataset size, plot the speedup of multi GPU over single GPU over the number of threads per block.
  #   Speedups are averaged over the different environment sizes (Ks).
  plt.figure(figsize=(12,6))
  for size in sizes:
    speedups_for_size = speedups_all[data_all[:,0]==size,:]
    avg_speedup_for_size = speedups_for_size.mean(axis=0)
    std_speedup_for_size = speedups_for_size.std(axis=0)
    plt.plot(threads_per_block, avg_speedup_for_size, label=f"size = {size:4.1f} [GiB]")
    plt.fill_between(threads_per_block, np.maximum(0,avg_speedup_for_size-std_speedup_for_size), avg_speedup_for_size+std_speedup_for_size, alpha=0.3)
    
  plt.title(f"Multi GPU speedups over single GPU\nDevice: {gpu_name}\n(averaged over environment sizes)")
  plt.xticks(threads_per_block)
  plt.xlabel("threads per block")
  plt.ylabel("speedup [x]")
  plt.legend()
  plt.savefig(f"{pc_name}_MultiGPU__Speedups_over_sizes.svg", dpi=600, bbox_inches="tight")
  plt.close()
  
  # =================================================================================================================== #
  
  # Plot 3: Throughputs for all combinations
  throughputs_single_gpu = data_all[:,0,np.newaxis] / (data_all[:,off+0::3] / 1000.)  # [GiB/s]
  throughputs_multi_gpu  = data_all[:,0,np.newaxis] / (data_all[:,off+1::3] / 1000.)  # [GiB/s]
  
  throughputs_single_gpu_avg = throughputs_single_gpu.mean(axis=1).reshape(sizes.size, Ks.size)
  throughputs_single_gpu_std = throughputs_single_gpu.std( axis=1).reshape(sizes.size, Ks.size)
  
  throughputs_multi_gpu_avg = throughputs_multi_gpu.mean(axis=1).reshape(sizes.size, Ks.size)
  throughputs_multi_gpu_std = throughputs_multi_gpu.std( axis=1).reshape(sizes.size, Ks.size)
  
  min_throughput = min(throughputs_single_gpu.min(), throughputs_multi_gpu.min())
  max_throughput = max(throughputs_single_gpu.max(), throughputs_multi_gpu.max())
  
  cmap   = plt.get_cmap("viridis")
  norm   = mpl.colors.Normalize(vmin=min_throughput, vmax=max_throughput, clip=True)
  mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
  scale  = 5
  alpha  = 0.7
  
  bubbles_single_gpu = [
    plt.Circle(
      xy     = (scale*K_ix, -scale*(size_ix+1)),
      radius = throughputs_single_gpu_std[size_ix,K_ix], 
      color  = mapper.to_rgba(throughputs_single_gpu_avg[size_ix,K_ix]), 
      alpha  = alpha
    )
    for K_ix in range(Ks.size)
    for size_ix in range(sizes.size)
  ]

  bubbles_multi_gpu = [
    plt.Circle(
      xy     = (scale*K_ix, scale*(size_ix+1)),
      radius = throughputs_multi_gpu_std[size_ix,K_ix], 
      color  = mapper.to_rgba(throughputs_multi_gpu_avg[size_ix,K_ix]), 
      alpha  = alpha
    )
    for K_ix in range(Ks.size)
    for size_ix in range(sizes.size)
  ]
  
  all_bubbles = bubbles_single_gpu + bubbles_multi_gpu

  fig, ax = plt.subplots(figsize=(12,6))
  ax.plot([-scale,scale*Ks.size],[0,0],'k--')
  
  for circ in all_bubbles:
    ax.add_patch(circ)
  
  ax.set_xlim([-scale,scale*Ks.size])
  ax.set_xticks(np.multiply(scale,range(Ks.size)), list(map(str,Ks)))
  ax.set_ylim([-scale*(sizes.size+1),scale*(sizes.size+1)])
  ys = np.multiply(scale, list(range(-sizes.size,0)) + list(range(1,sizes.size+1)))
  ax.set_yticks(ys, [f"{s:4.1f}" for s in sizes.tolist()[::-1]+sizes.tolist()])
  ax.set_aspect('equal')
  
  fig.colorbar(mapper, ax=ax, ticks=[min_throughput, max_throughput], format="%4.1f")
  
  ax.set_title(f"Throughput [GiB/s]\nDevice: {gpu_name}")
  ax.set_ylabel("Dataset size [GiB]")
  ax.set_xlabel("Environment size")
  ax.text(-scale*3.1, scale*sizes.size+3,"Multi GPU")
  ax.text(-scale*3.1,-scale*sizes.size-4,"Single GPU")
  
  plt.savefig(f"{pc_name}_MultiGPU__Throughputs.svg", dpi=600, bbox_inches="tight")
  plt.close()
  

if __name__ == '__main__':
  import sys 
  if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} <gpu-name> <pc-name> <multi-GPU.csv> <has-single-threaded:y/n>")
    exit(1)
  
  plot_multi_GPU(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]=='y')
  