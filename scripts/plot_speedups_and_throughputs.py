
import os
import re
import tqdm
import numpy as np
import matplotlib.pyplot as plt

BENCHMARK_TYPES = ['CUDA vs OpenCL', 'CUDA Allocation Size']

def parse_files(csvs):
  def __get_benchmark_type(csv_filename):
    base_name = os.path.basename(csv_filename)
    types_and_patterns = [
      (BENCHMARK_TYPES[0], re.compile(r'^benchmark_maskedProcessing_([a-zA-Z0-9]+).csv$')),
      (BENCHMARK_TYPES[1], re.compile(r'^benchmark_cudaBiggerAllocation_([a-zA-Z0-9]+).csv$')),
    ]
    for _type, _pattern in types_and_patterns:
      _match = _pattern.match(base_name)
      if _match is not None:
        return _type, _match.group(1)
    raise ValueError("Error: Unrecognized benchmark type.")

  def __parse_line(line):
    # A line always consists of ...
    #   o ... the dataset size in GiB (floating point),
    #   o ... the CPU parts (2 times a pair of the environment size and the execution times in ms),
    #   o ... pairs of the number of threads per block (integer) and the execution time (in ms, floating point).
    __parts = line.split(',')
    try:
      seq_time = float(__parts[2])
    except Exception:
      seq_time = -1
    
    return {
      'size_in_GiB': float(__parts[0]),
      'K'          : int(__parts[1]),
      'cpu': {
        'sequential': seq_time,
        'OpenMP'    : float(__parts[4])
      },
      'gpu': [ (int(__parts[i]), float(__parts[i+1])) for i in range(5, len(__parts), 2) ]
    }

  configurations = []
  
  for csv_filename in csvs:
    with open(csv_filename, "r") as csv:
      benchmark_type, device_name = __get_benchmark_type(csv_filename)
      config = {
        'device_name'    : device_name,
        'benchmark_type' : benchmark_type,
        'executions'     : []
      }
    
      for line in csv:
        if line[0] == '=':
          continue
        config['executions'].append(__parse_line(line))
      
      configurations.append(config)
  
  return configurations


# -------------------------------------------------------------------------------------------------

def extract(key, configuration, cond=lambda x: True):
  __access = {
    'Ks'         : lambda e: e['K'],
    'sizes'      : lambda e: e['size_in_GiB'],
    'sequential' : lambda e: e['cpu']['sequential'],
    'OpenMP'     : lambda e: e['cpu']['OpenMP'],
    'gpu_times'  : lambda e: [ _time for _, _time in e['gpu'] ]
  }
  return np.array([ __access[key](e) for e in configuration['executions'] if cond(e) ])
  

def plot_over_threads_per_block(gpu_name, configuration, selection_criterion, computation_criterion, baseline):
  threads_per_block = np.unique([ tpb for tpb, _ in configuration['executions'][0]['gpu'] ])
  dataset_sizes     = np.unique(extract('sizes', configuration))
  Ks                = np.unique(extract('Ks',    configuration))

  averaging_over = {
    'K'    : 'sizes',
    'sizes': 'Ks'
  }[selection_criterion]

  data_group = {
    'K'    : Ks,
    'sizes': dataset_sizes
  }[selection_criterion]
  
  line_label = {
    'K'    : lambda k: f"K = {k}",
    'sizes': lambda s: f"Size = {np.around(s, decimals=1)} GiB"
  }[selection_criterion]
  
  selection_function_t = {
    'K'    : lambda e, val: e['K'] == val,
    'sizes': lambda e, val: abs(e['size_in_GiB'] - val) < 1e-5
  }[selection_criterion]
  
  fill_std = {
    'K'    : True,
    'sizes': False
  }[selection_criterion]
  
  computation_function = {
    'speedup'   : lambda baseline, exec_times,      _: np.true_divide(baseline[:,np.newaxis], exec_times),
    'throughput': lambda        _, exec_times, memory: np.true_divide(  memory[:,np.newaxis], exec_times / 1000.0)
  }[computation_criterion]
  
  computation_unit = {
    'speedup'   : 'x',
    'throughput': 'GiB/s'
  }[computation_criterion]
  
  title_infix = {
    'speedup'   : f"over {baseline} CPU execution",
    'throughput': ""
  }[computation_criterion]
  
  plt.figure(figsize=(15,10))
  
  min_value, max_value = float('inf'), 0
  for value in data_group:
    selection_function = lambda e: selection_function_t(e, value)
    baseline_time = extract(baseline,    configuration, selection_function)
    exec_times    = extract('gpu_times', configuration, selection_function)
    sizes         = extract('sizes',     configuration, selection_function)
    
    computed_values     = computation_function(baseline_time, exec_times, sizes)
    
    # If we have a multi-GPU setup, more values are measured. Average them to get a good estimate.
    values_reshaped = computed_values.reshape((baseline_time.size, -1, threads_per_block.size))
    values_per_device = np.mean(values_reshaped, axis=1)
    
    computed_values_avg = np.mean(values_per_device, axis=0)
    computed_values_std = np.std( values_per_device, axis=0)
    
    min_value = min(min_value, np.min(computed_values_avg - computed_values_std) if fill_std else np.min(computed_values_avg) )
    max_value = max(max_value, np.max(computed_values_avg + computed_values_std) if fill_std else np.max(computed_values_avg) )

    plt.plot(threads_per_block, computed_values_avg, label=line_label(value))
    if fill_std:
      plt.fill_between(threads_per_block, np.maximum(computed_values_avg-computed_values_std, 0), computed_values_avg+computed_values_std, alpha=0.25)
  
  
  plt.title(f"CUDA {computation_criterion}s {title_infix}\n(device: {gpu_name}; averaged over {averaging_over})")
  plt.xticks(threads_per_block)
  plt.ylim([max(min_value * 0.95, 0), max_value * 1.05])
  plt.xlabel("threads/block")
  plt.ylabel(f"{computation_criterion} [{computation_unit}]")
  plt.legend()
  plt.savefig(f"{configuration['device_name']}_{computation_criterion}_over_{baseline}_{selection_criterion}_{'_'.join(configuration['benchmark_type'].split())}.svg", dpi=600)
  plt.close()

def plot_CUDA_over_threads_per_block(gpu_name, configurations):
  for configuration in configurations:
    for baseline in ['sequential', 'OpenMP']:
      for computation_criterion in ['speedup', 'throughput']:
        plot_over_threads_per_block(gpu_name, configuration, 'K',     computation_criterion, baseline)
        plot_over_threads_per_block(gpu_name, configuration, 'sizes', computation_criterion, baseline)


def plot_CUDA_one_size_over_threads_per_block(gpu_name, configurations):
  def __plot_multiple_K_lines(configuration, computation_criterion, baseline):
    computation_function = {
      'speedup'   : lambda baseline, exec_times,      _: np.true_divide(baseline[:,np.newaxis], exec_times),
      'throughput': lambda        _, exec_times, memory: np.true_divide(  memory[:,np.newaxis], exec_times / 1000.0)
    }[computation_criterion]
    
    computation_unit = {
      'speedup'   : 'x',
      'throughput': 'GiB/s'
    }[computation_criterion]

    title_infix = {
      'speedup'   : f"over {baseline} CPU execution",
      'throughput': ""
    }[computation_criterion]
    
    threads_per_block = np.unique([ tpb for tpb, _ in configuration['executions'][0]['gpu'] ])
    Ks                = np.unique(extract('Ks',    configuration))
    target_size       = np.min(extract('sizes', configuration)) # only do the plot for the smallest size

    plt.figure(figsize=(15,10))
    
    min_value, max_value = float('inf'), 0
    for K in Ks:
      selection_function = lambda e: abs(e['size_in_GiB'] - target_size) < 1e-5 and e['K'] == K
      baseline_time = extract(baseline,    configuration, selection_function)
      exec_times    = extract('gpu_times', configuration, selection_function)
      sizes         = extract('sizes',     configuration, selection_function)
      
      computed_values = computation_function(baseline_time, exec_times, sizes)[0,...]

      # account for multi-GPU setups
      values_reshaped = computed_values.reshape((baseline_time.size, -1, threads_per_block.size))
      values_per_device = np.mean(values_reshaped, axis=1) # (1 x threads/block)

      min_value = min(min_value, np.min(values_per_device))
      max_value = max(max_value, np.max(values_per_device))
      
      plt.plot(threads_per_block, values_per_device[0,...], label=f"K = {K}")
    
    plt.title(f"CUDA {computation_criterion}s {title_infix}\n(device: {gpu_name}; processing {np.around(target_size, decimals=1)} GiB)")
    plt.xticks(threads_per_block)
    plt.ylim([max(min_value * 0.95, 0), max_value * 1.05])
    plt.xlabel("threads/block")
    plt.ylabel(f"{computation_criterion} [{computation_unit}]")
    plt.legend()
    plt.savefig(f"{configuration['device_name']}_{computation_criterion}_over_{baseline}_singleSize_{'_'.join(configuration['benchmark_type'].split())}.svg", dpi=600)
    plt.close()

  for configuration in configurations:
    for baseline in ['sequential', 'OpenMP']:
      for computation_criterion in ['speedup', 'throughput']:
        __plot_multiple_K_lines(configuration, computation_criterion, baseline)


# -------------------------------------------------------------------------------------------------

def plot_comparison(gpu_name, configurations):
  def __plot_comparison(configuration, num_ocl_devices, selection_criterion, computation_criterion):
    threads_per_block = np.unique([ tpb for tpb, _ in configuration['executions'][0]['gpu'] ])
    dataset_sizes     = np.unique(extract('sizes', configuration))
    Ks                = np.unique(extract('Ks',    configuration))

    averaging_over = {
      'K'    : 'sizes',
      'sizes': 'Ks'
    }[selection_criterion]

    data_group = {
      'K'    : Ks,
      'sizes': dataset_sizes
    }[selection_criterion]
    
    line_label = {
      'K'    : lambda k: f"K = {k}",
      'sizes': lambda s: f"Size = {np.around(s, decimals=1)} GiB"
    }[selection_criterion]
    
    selection_function_t = {
      'K'    : lambda e, val: e['K'] == val,
      'sizes': lambda e, val: abs(e['size_in_GiB'] - val) < 1e-5
    }[selection_criterion]
    
    computation_function = {
      'speedup'   : lambda baseline, exec_times,      _: np.true_divide(baseline, exec_times),
      'throughput': lambda        _, exec_times, memory: np.true_divide(  memory, exec_times / 1000.0)
    }[computation_criterion]
    
    computation_unit = {
      'speedup'   : 'x',
      'throughput': 'GiB/s'
    }[computation_criterion]
    
    title_infix = {
      'speedup'   : 'relative to sequential CPU execution; ',
      'throughput': ''
    }[computation_criterion]
    
    def __to_coord(data_idcs, group_idx, bar_width = 0.2): # data_idx == 1 for seq, 2 for omp, ...;  group_idx == idx of subgroup
      off = (group_idx - (threads_per_block.size - 1) // 2) * bar_width
      return [ 2*data_idx + off for data_idx in data_idcs ]

    plt.figure(figsize=(15,10))
    
    for data_idx, value in enumerate(data_group):
      
      selection_function = lambda e: selection_function_t(e, value)
      sizes   = extract('sizes',      configuration, selection_function)
      cpu_seq = extract('sequential', configuration, selection_function) # (group x 1)
      cpu_omp = extract('OpenMP',     configuration, selection_function) # (group x 1)
      gpu     = extract('gpu_times',  configuration, selection_function) # (group x (nr_cuda + nr_ocl) x threads_per_block_without_cuda_optimal)
      
      nr_ocl  = num_ocl_devices
      nr_cuda = gpu.shape[1] // (threads_per_block.size-1) - nr_ocl
      # cuda : (group x (nr_cuda * threads_per_block_without_cuda_optimal))
      cuda    = np.concatenate([ gpu[:, i*threads_per_block.size : i*threads_per_block.size+(threads_per_block.size-1)] for i in range(nr_cuda) ]).reshape((cpu_seq.shape[0], -1))
      # ocl : (group x (nr_ocl * threads_per_block_without_cuda_optimal))
      ocl     = gpu[:, nr_cuda*threads_per_block.size:]
      
      # Compute all the data (speedup/throughput) compared to the sequential CPU execution 
      seq_improvement  = computation_function(cpu_seq,               cpu_seq, sizes)                 # (group x 1)
      omp_improvement  = computation_function(cpu_seq,               cpu_omp, sizes)                 # (group x 1)
      ocl_improvement  = computation_function(cpu_seq[:,np.newaxis], ocl,     sizes[:,np.newaxis])   # (group x (nr_ocl  x threads_per_block_without_cuda_optimal)
      cuda_improvement = computation_function(cpu_seq[:,np.newaxis], cuda,    sizes[:,np.newaxis])   # (group x (nr_cuda x threads_per_block_without_cuda_optimal)
      
      improvements     = [ seq_improvement[:,np.newaxis], omp_improvement[:,np.newaxis], ocl_improvement, cuda_improvement ]

      
      # For current group, e.g., the current value of K, average over the other property (e.g., size) AND the threads per block (!)
      x = __to_coord(range(len(improvements)), data_idx) # x-coord for each "value" of "data_group", e.g., each K in Ks
      y    = [ np.mean(impr) for impr in improvements ]
      yerr = [  np.std(impr) for impr in improvements ]
      plt.errorbar(x, y, yerr=yerr, capsize=6, fmt="o", label=line_label(value))

    if computation_criterion == 'speedup':
      _xmin, _xmax = plt.xlim()
      plt.plot([_xmin, _xmax], [1, 1], '--k')
    
    plt.title(f"Comparison of {computation_criterion}s over different devices \n({title_infix}averaged over {averaging_over})")
    plt.xticks(__to_coord(range(4), len(data_group)//2), ['Sequential', 'OpenMP', 'OpenCL', 'CUDA'])
    plt.legend()
    plt.xlabel("execution")
    plt.ylabel(f"{computation_criterion} [{computation_unit}]")
    plt.savefig(f"{configuration['device_name']}_{computation_criterion}_{selection_criterion}_{'_'.join(configuration['benchmark_type'].split())}.svg", dpi=600)
    plt.close()

  for configuration in configurations:
    num_ocl_devices = int(input("Number of OpenCL devices: "))
    __plot_comparison(configuration, num_ocl_devices, 'K',     'speedup')
    __plot_comparison(configuration, num_ocl_devices, 'sizes', 'speedup')
    __plot_comparison(configuration, num_ocl_devices, 'K',     'throughput')
    __plot_comparison(configuration, num_ocl_devices, 'sizes', 'throughput')


# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  import sys
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <gpu-name> <path-to-csv> [<path-to-csv>...]")
    exit(1)

  gpu_name = sys.argv[1]
  configurations = parse_files(sys.argv[2:])
  
  # 1. plot the speedups and the throughputs for the CUDA case only
  print("Plotting the CUDA speedups and throughputs                                    ... ", end="")
  relevant_configurations = filter(lambda conf: conf['benchmark_type'] == BENCHMARK_TYPES[1], configurations)
  plot_CUDA_over_threads_per_block(gpu_name, relevant_configurations)
  print("done.")
  
  # 2. plot the speedups and throughputs without averaging, i.e., plot multiple lines (one for each K), for the smallest size only (CUDA only)
  print("Plotting the speedups and throughputs for the smallest size only              ... ", end="")
  relevant_configurations = filter(lambda conf: conf['benchmark_type'] == BENCHMARK_TYPES[1], configurations)
  plot_CUDA_one_size_over_threads_per_block(gpu_name, relevant_configurations)
  print("done.")
  
  # 3. plot the OpenCL vs. CUDA vs. CPU comparison
  print("Plotting the comparison of CPU vs. OpenCL vs. CUDA ... ")
  comparison_configs = filter(lambda conf: conf['benchmark_type'] == BENCHMARK_TYPES[0], configurations)
  plot_comparison(gpu_name, comparison_configs)
  print("Plotting the comparison of CPU vs. OpenCL vs. CUDA ... done.")
  

