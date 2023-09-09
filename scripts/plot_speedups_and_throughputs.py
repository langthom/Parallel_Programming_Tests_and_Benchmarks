
import os
import re
import numpy as np
import matplotlib.pyplot as plt

BENCHMARK_TYPES = ['CUDA vs OpenCL', 'CUDA Allocation Size', 'CUDA Multi-GPU']

BASE_LINE_METHOD = 'OpenMP'

def parse_files(csvs):
  def __get_benchmark_type(csv_filename):
    base_name = os.path.basename(csv_filename)
    types_and_patterns = [
      (BENCHMARK_TYPES[0], re.compile('^benchmark_maskedProcessing_(\w+).csv$')),
      (BENCHMARK_TYPES[1], re.compile('^benchmark_cudaBiggerAllocation_(\w+).csv$')),
      (BENCHMARK_TYPES[2], re.compile('^benchmark_cudaMultiGPU_(\w+).csv$'))
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
    'Ks'       : lambda e: e['K'],
    'sizes'    : lambda e: e['size_in_GiB'],
    'baseline' : lambda e: e['cpu'][BASE_LINE_METHOD],
    'gpu_times': lambda e: [ _time for _, _time in e['gpu'] ]
  }
  return np.array([ __access[key](e) for e in configuration['executions'] if cond(e) ])
  

def plot_over_threads_per_block(gpu_name, configuration, selection_criterion, computation_criterion):
  threads_per_block = [ tpb for tpb, _ in configuration['executions'][0]['gpu'] ]
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
  
  
  plt.figure(figsize=(15,10))
  
  min_value, max_value = float('inf'), 0
  for value in data_group:
    selection_function = lambda e: selection_function_t(e, value)
    baseline_time = extract('baseline',  configuration, selection_function)
    exec_times    = extract('gpu_times', configuration, selection_function)
    sizes         = extract('sizes',     configuration, selection_function)
    
    computed_values     = computation_function(baseline_time, exec_times, sizes)
    computed_values_avg = np.mean(computed_values, axis=0)
    computed_values_std = np.std( computed_values, axis=0)
    
    min_value = min(min_value, np.min(computed_values_avg - computed_values_std) if fill_std else np.min(computed_values_avg) )
    max_value = max(max_value, np.max(computed_values_avg + computed_values_std) if fill_std else np.max(computed_values_avg) )

    plt.plot(threads_per_block, computed_values_avg, label=line_label(value))
    if fill_std:
      plt.fill_between(threads_per_block, np.maximum(computed_values_avg-computed_values_std, 0), computed_values_avg+computed_values_std, alpha=0.25)
  
  
  plt.title(f"CUDA {computation_criterion}s over {BASE_LINE_METHOD} CPU execution\n(device: {gpu_name}; averaged over {averaging_over})")
  plt.xticks(threads_per_block)
  plt.ylim([max(min_value - 5, 0), max_value + 5])
  plt.xlabel("threads/block")
  plt.ylabel(f"{computation_criterion} [{computation_unit}]")
  plt.legend()
  plt.savefig(f"{configuration['device_name']}_{computation_criterion}_{selection_criterion}_{'_'.join(configuration['benchmark_type'].split())}.pdf", dpi=300)


def plot_CUDA_over_threads_per_block(gpu_name, configurations):
  for configuration in configurations:
    plot_over_threads_per_block(gpu_name, configuration, 'K',     'speedup')
    plot_over_threads_per_block(gpu_name, configuration, 'sizes', 'speedup')
    plot_over_threads_per_block(gpu_name, configuration, 'K',     'throughput')
    plot_over_threads_per_block(gpu_name, configuration, 'sizes', 'throughput')


# -------------------------------------------------------------------------------------------------

def pair_up_multiGPU_configurations(configurations):
  # yield pairs of configurations of a multi GPU and a single GPU benchmark of a single PC
  for config in configurations:
    if config['benchmark_type'] == BENCHMARK_TYPES[2]:
      single_gpu_config = [ c for c in configurations if c['benchmark_type'] == BENCHMARK_TYPES[1] and c['device_name'] == config['device_name'] ]
      yield (config, single_gpu_config[0])


def plot_multiGPU_speedups(gpu_name, configuration_pairs):
  def __plot_multiGPU(multi_GPU_config, single_gpu_config, selection_criterion):
    threads_per_block = [ tpb for tpb, _ in multi_GPU_config['executions'][0]['gpu'] ]
    dataset_sizes     = np.unique(extract('sizes', multi_GPU_config))
    Ks                = np.unique(extract('Ks',    multi_GPU_config))

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

    plt.figure(figsize=(15,10))
    min_value, max_value = float('inf'), 0
    for value in data_group:
      selection_function = lambda e: selection_function_t(e, value)
      baseline_time = extract('gpu_times', single_gpu_config, selection_function) # (group x threads/block), baseline now is the gpu speeds of the single GPU version ...
      exec_times    = extract('gpu_times',  multi_GPU_config, selection_function) # (group x threads/block), ... for a comparison with the multi GPU setup
      
      computed_values     = np.true_divide(baseline_time, exec_times)
      computed_values_avg = np.mean(computed_values, axis=0)
      computed_values_std = np.std( computed_values, axis=0)
      
      min_value = min(min_value, np.min(computed_values_avg - computed_values_std) if fill_std else np.min(computed_values_avg) )
      max_value = max(max_value, np.max(computed_values_avg + computed_values_std) if fill_std else np.max(computed_values_avg) )

      plt.plot(threads_per_block, computed_values_avg, label=line_label(value))
      if fill_std:
        plt.fill_between(threads_per_block, np.maximum(computed_values_avg-computed_values_std, 0), computed_values_avg+computed_values_std, alpha=0.25)
    
    print(f"min/max: {min_value} / {max_value}")
    plt.title(f"Multi-GPU speedups (CUDA) over single GPU execution\n(device: {gpu_name}; averaged over {averaging_over})")
    plt.xticks(threads_per_block)
    plt.ylim([max(min_value * 0.95, 0), max_value * 1.05])
    plt.xlabel("threads/block")
    plt.ylabel(f"speedup [x]")
    plt.legend()
    plt.savefig(f"{multi_GPU_config['device_name']}_speedup_{selection_criterion}_{'_'.join(multi_GPU_config['benchmark_type'].split())}.pdf", dpi=300)

  for multi_gpu_config, single_gpu_config in configuration_pairs:
    __plot_multiGPU(multi_gpu_config, single_gpu_config, 'K')
    __plot_multiGPU(multi_gpu_config, single_gpu_config, 'sizes')


# -------------------------------------------------------------------------------------------------

def plot_comparison(configurations):
  pass

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  import sys
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <gpu-name> <path-to-csv> [<path-to-csv>...]")
    exit(1)

  gpu_name       = sys.argv[1]
  configurations = parse_files(sys.argv[2:])
  
  # 1. plot the speedups and the throughputs for the CUDA case only
  relevant_configurations = filter(lambda conf: conf['benchmark_type'] == BENCHMARK_TYPES[1], configurations)
  #plot_CUDA_over_threads_per_block(gpu_name, relevant_configurations)
  
  # 2. multi GPU case:
  #    2.1 plot the speeups of multi GPU execution over the single gpu (bigger allocation) case
  multi_GPU_speedup_config_pairs = pair_up_multiGPU_configurations(configurations)
  plot_multiGPU_speedups(gpu_name, multi_GPU_speedup_config_pairs)
  
  #    2.2 plot the individual speedups and throughputs of the multi GPU case
  multi_gpu_configs = filter(lambda conf: conf['benchmark_type'] == BENCHMARK_TYPES[2], configurations)
  #plot_CUDA_over_threads_per_block(gpu_name, multi_gpu_configs)

