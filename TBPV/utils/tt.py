# l = ['000011', '000001', '000001']
# for i, j, k in zip(*l):
#     print(i, j, k)

# print(''.join(i + j + k for i, j, k in zip(*l)))

import numpy as np

block_size = 64
points = [[63, 67, 150, 3, 3, 2], [3, 57, 180, 3 ,4, 2], [100, 4, 160, 6, 34, 3], [34, 79, 170, 23, 43, 2]]
points = np.asarray(points)
block_ids = points[:, :3] // block_size
block_ids = block_ids.astype(np.uint32)

# block_ids = [[0, 1, 2], [0, 0, 2], [1, 0, 2], [0, 1, 2]]
print('---------block_ids--------')
print(block_ids)
block_ids_unique, block_idx, block_len = np.unique(block_ids, return_inverse=True, return_counts=True, axis=0)
print('---------block_ids_unique--------')
print(block_ids_unique)
print('---------block_idx--------')
print(block_idx)
print('---------block_len--------')
print(block_len)

sort_key = []
for x, y, z in block_ids_unique:
    zip_params = [f'{v:06b}' for v in [z, y, x]]#output: 1008b, 1308b... 10,13 are z and y,
    print('------zipparams---------')
    print(zip_params)
    sort_key.append(''.join(i + j + k for i, j, k in zip(*zip_params)))
    print('------sort key---------')
    print(sort_key)
sort_idx = np.argsort(sort_key) 
print('------sort_idx---------')
print(sort_idx)

block_ids_unique = block_ids_unique[sort_idx]
print('---------block_ids_unique--------')
print(block_ids_unique)
block_len = block_len[sort_idx]
print('---------block_len--------')
print(block_len)
# invert permutation
inv_sort_idx = np.zeros_like(sort_idx)
print('--------inv sort idx--------')
print(inv_sort_idx)
inv_sort_idx[sort_idx] = np.arange(sort_idx.size)
print('--------inv sort idx--------')
print(inv_sort_idx)
block_idx = inv_sort_idx[block_idx]
print('---------block_idx--------')
print(block_idx)

# Translate points into local block coordinates
local_refs = np.pad(block_ids_unique[block_idx] * block_size, [[0, 0], [0, points.shape[1] - 3]])#add padding in block ids unique, to have the same size as points
points_local = points - local_refs#rotate,translate original coordinatte to box coordinate by subtract the reference
print('---------local_refs--------')
print(local_refs)
print('---------points_local--------')
print(points_local)

# Group points by block
blocks = [np.zeros((l, points.shape[1])) for l in block_len]#blocks: l in block len:number of point are assign to the same block
print('---------blocks--------')
print(blocks)
#points shape[1]=6. (x,y,z and 3 colors)
blocks_last_idx = np.zeros(len(block_len), dtype=np.uint32)
print('---------blocks last idx--------')
print(blocks_last_idx)
for i, b_idx in enumerate(block_idx):
    blocks[b_idx][blocks_last_idx[b_idx]] = points_local[i]#store point coordinate in blocks
    blocks_last_idx[b_idx] += 1#increase counter by 1

print('---------blocks--------')
print(blocks)

print('---------blocks last idx--------')
print(blocks_last_idx)

