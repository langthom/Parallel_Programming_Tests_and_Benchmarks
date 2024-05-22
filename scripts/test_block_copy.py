
from math import ceil
import numpy as np

def num_tiles(dim, tile_size):
  return int( ceil( float(dim) / tile_size ) )

def test_block_copy(volume_shape, block_shape, K):
  K2 = K // 2
  pad = K - 1
  
  input_volume     = np.random.randint(0, 65530, volume_shape, dtype=int)
  output_volume    = np.full([s - pad for  s in volume_shape], fill_value=-1, dtype=input_volume.dtype)

  full_block_shape = [ bs + K - 1 for bs in block_shape ]
  extracted_block  = np.full(full_block_shape, fill_value=-1, dtype=input_volume.dtype)

  num_block_tiles  = [ num_tiles( full_block_shape[i], block_shape[i] ) for i in range(3) ]
  
  # simulate execution of 3D blocks
  n_blocks = [ num_tiles(volume_shape[i], block_shape[i]) for i in range(3) ]
  
  # block launch loops (not present in device code)
  for block_z in range(n_blocks[0]):
    for block_y in range(n_blocks[1]):
      for block_x in range(n_blocks[2]):
        # assignment loops (not present in device code)
        for tz in range(block_shape[0]):
          for ty in range(block_shape[1]):
            for tx in range(block_shape[2]):
            
              # tiling loops
              for block_tile_z in range(num_block_tiles[0]):
                for block_tile_y in range(num_block_tiles[1]):
                  for block_tile_x in range(num_block_tiles[2]):
                    
                    z_within_full_block = block_tile_z * block_shape[0] + tz
                    y_within_full_block = block_tile_y * block_shape[1] + ty
                    x_within_full_block = block_tile_x * block_shape[2] + tx
                    
                    gz = block_z * block_shape[0] + z_within_full_block
                    gy = block_y * block_shape[1] + y_within_full_block
                    gx = block_x * block_shape[2] + x_within_full_block
                    
                    if np.less([z_within_full_block,y_within_full_block,x_within_full_block], full_block_shape).all() and np.less([gz,gy,gx], volume_shape).all():
                      extracted_block[z_within_full_block,y_within_full_block,x_within_full_block] = input_volume[gz,gy,gx]

        # Process the current block
        processed_block = np.zeros_like(extracted_block)
        for tz in range(block_shape[0]):
          for ty in range(block_shape[1]):
            for tx in range(block_shape[2]):
              z = tz + K2
              y = ty + K2
              x = tx + K2
              processed_block[z,y,x] = extracted_block[z-K2:z+K2+1,y-K2:y+K2+1,x-K2:x+K2+1].mean()
        
        # Write back processed chunk
        for tz in range(block_shape[0]):
          for ty in range(block_shape[1]):
            for tx in range(block_shape[2]):
              gz = block_z * block_shape[0] + tz
              gy = block_y * block_shape[1] + ty
              gx = block_x * block_shape[2] + tx
              
              if np.less([gz,gy,gx], output_volume.shape).all():
                output_volume[gz,gy,gx] = processed_block[tz+K2,ty+K2,tx+K2]

  # Compare to the ground truth
  ground_truth = np.zeros_like(output_volume)
  for i, _ in np.ndenumerate(ground_truth):
    ii = tuple( _i + K2 for _i in i )
    ground_truth[i] = input_volume[tuple(slice(ii[j]-K2,ii[j]+K2+1,None) for j in range(3))].mean()
  
  
  print(f"ERR: {np.linalg.norm(ground_truth - output_volume)}")



if __name__ == '__main__':
  test_block_copy((50,50,50), (4,8,32), K=3)
  test_block_copy((10,20,50), (4,8,32), K=3)
  test_block_copy((50,50,50), (4,8,32), K=5)
  test_block_copy((10,20,50), (4,8,32), K=5)
