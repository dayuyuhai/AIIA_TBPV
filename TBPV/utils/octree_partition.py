import numpy as np
def compute_new_bbox(idx, bbox_min, bbox_max):
    midpoint = (bbox_max - bbox_min) // 2 + bbox_min
    # Compute global block bounding box
    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()
    # print('----------midpoint-----------')
    # print(midpoint)
    # print('----------cur_bbox_min-----------')
    # print(cur_bbox_min)
    # print('----------cur_bbox_max-----------')
    # print(cur_bbox_max)
    if idx & 1:
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]
    # print('----------2 cur_bbox_min-----------')
    # print(cur_bbox_min)
    # print('----------2 cur_bbox_max-----------')
    # print(cur_bbox_max)
#idx=0 ->> lowest box, id=1--> next box..., this function split a box bounded in bbox_min qnd bbox_max(3 dimension)
    return cur_bbox_min, cur_bbox_max


# Partitions points using an octree scheme
# returns points in local coordinates (block) and octree structure as an 8 bit integer (right to left order)
def split_octree(points, bbox_min, bbox_max):
    ret_points = [[] for x in range(8)]
    print('-------ret point---------')
    print(ret_points)
    midpoint = (bbox_max - bbox_min) // 2
    print('----------midpoint-----------')
    print(midpoint)

    global_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]
    print('----------global_bboxes-----------')
    print(global_bboxes)    
    # Translate into local block coordinates
    # Use local block bounding box
    local_bboxes = [(np.zeros(3), x[1] - x[0]) for x in global_bboxes]
    print('----------local_bboxes-----------')
    print(local_bboxes)     
    for point in points:
        location = 0
        print('---------point------')
        print(point)
        if point[0] >= midpoint[0]:
            location |= 0b001
        if point[1] >= midpoint[1]:
            location |= 0b010
        if point[2] >= midpoint[2]:
            location |= 0b100
        # print('-----------location------------')
        # print(location)
        ret_points[location].append(point - np.pad(global_bboxes[location][0], [0, len(point) - 3]))
        print('-----------retpoint------------')
        print(ret_points)
    binstr = 0b00000000
    for i, rp in enumerate(ret_points):
        if len(rp) > 0:
            binstr |= (0b00000001 << i)
    print('--------bit str---------')
    print(binstr)
    '''
        np.vstack:
            ??????????????????????????????????????????????????????????????????
            ??????????????????????????????????????????
    '''
    aa = [np.vstack(rp) for rp in ret_points if len(rp) > 0]
    print('---------return thing----------')
    print(aa)
    return [np.vstack(rp) for rp in ret_points if len(rp) > 0], binstr, local_bboxes


# Returns list of blocks and octree structure as a list of 8 bit integers
# Recursive octree partitioning function that is slow for high number of points (> 500k)
def partition_octree_rec(points, bbox_min, bbox_max, level):
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    print("-------begin split octree ????????????????????????????????????,bitstr????????????????????????,bbox??????block??????, retpoint?????????block?????????????????????------------")
    ret_points, binstr, bboxes = split_octree(points, bbox_min, bbox_max)
    print("-------split octree end--------------")
    result = [partition_octree(rp, bbox[0], bbox[1], level - 1) for rp, bbox in zip(ret_points, bboxes)]
    print("-------------result shape and result--------------")
    print(np.asarray(result).shape)
    print(result)
    blocks = [subblock for block_res in result for subblock in block_res[0] if len(subblock) > 0]
    print("----blocks after digui-----------")
    print(blocks)
    new_binstr = [binstr] + [subbinstr for block_res in result if block_res[1] is not None for subbinstr in block_res[1]]
    print("----------new binstr------------")
    print(new_binstr)
    return blocks, new_binstr


# Returns list of blocks and octree structure as a list of 8 bit integers
# This version should be much faster than the fully recursive version
# Example: longdress_vox10 73.6s for recursive versus 7.6s for iterative
# However, this assumes that bbox_min is [0, 0, 0] and bbox_max is a power of 2
# sample command : timing(partition_octree)(points, [0, 0, 0], [1024, 1024, 1024], 4)
def partition_octree(points, bbox_min, bbox_max, level):
    points = np.asarray(points)
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None
    # print('----------arrpoints ??????+??????---------')
    # print(points)
    bbox_min = np.asarray(bbox_min)
    # print('------------boxmin-------------')
    # print(bbox_min)
    np.testing.assert_array_equal(bbox_min, [0, 0, 0])
    bbox_max = np.asarray(bbox_max)
    geo_level = int(np.ceil(np.log2(np.max(bbox_max))))
    # print('-----geolevel--------')
    # print(geo_level)
    assert geo_level >= level#check whether geo level larger than level, if not, raise an error
    block_size = 2 ** (geo_level - level)#high level value --> smaller block size
    # print('-----------blocksize----------')
    # print(block_size)

    '''
        np.unique():??????????????????????????????????????????
            1. return_inverse???True???:??????????????????????????????????????????????????????indices
            2. return_count???True???:???????????????????????????????????????????????? counts
    '''
    # Compute partitions for each point
    # ????????????????????????????????????
    block_ids = points[:, :3] // block_size #floor division, like quantization
    block_ids = block_ids.astype(np.uint32)
    # print('-----------blockid ????????????+??????blocksize(??????)----------')
    # print(block_ids, block_ids.shape)
    block_ids_unique, block_idx, block_len = np.unique(block_ids, return_inverse=True, return_counts=True, axis=0)
    #block ids unique: unique triple sets of index in x,y,x direction
    #block idx : index to reverse to the block-ids from block ids unique
    #block-len: number of occurence of each element in block ids unique
    # print('---------block_ids_unique ???????????????+??????--------')
    # print(block_ids_unique, block_ids_unique.shape)
    # print('---------block_idx ??????????????????????????????????????????????????????indices--------')
    # print(block_idx)
    # print('---------block_len ??????????????????????????????????????????--------')
    # print(block_len)

    # Interleave coordinate bits to reorder by octree block order
    sort_key = []
    for x, y, z in block_ids_unique:
        zip_params = [f'{v:0{geo_level - level}b}' for v in [z, y, x]]#output: 1008b, 1308b... 10,13 are z and y,
        # print('------zipparams block_ids_unique??????????????????---------')
        # print(zip_params)
        sort_key.append(''.join(i + j + k for i, j, k in zip(*zip_params)))
        # print('------sort key---------')
        # print(sort_key)
    sort_idx = np.argsort(sort_key) # ??????+??????????????????????????????????????????????????????
    # print('------sort_idx---------')
    # print(sort_idx)
    block_ids_unique = block_ids_unique[sort_idx]
    # print('---------block_ids_unique ?????????????????????????????????--------')
    # print(block_ids_unique)
    block_len = block_len[sort_idx]
    # print('---------block_len ????????????block_ids_unique--------')
    # print(block_len) # ?????????????????????185???
    # invert permutation
    inv_sort_idx = np.zeros_like(sort_idx)
    # print('--------inv sort idx--------')
    # print(inv_sort_idx)
    inv_sort_idx[sort_idx] = np.arange(sort_idx.size)
    block_idx = inv_sort_idx[block_idx]
    # print('--------inv sort idx--------')
    # print(inv_sort_idx)
    # print('---------block_idx ????????????block_ids_unique--------')
    # print(block_idx)

    # Translate points into local block coordinates
    local_refs = np.pad(block_ids_unique[block_idx] * block_size, [[0, 0], [0, points.shape[1] - 3]])#add padding in block ids unique, to have the same size as points
    points_local = points - local_refs#rotate,translate original coordinatte to box coordinate by subtract the reference
    # print('---------local_refs ???????????? + shape?????????????????????0??????--------')
    # print(local_refs)
    # print('---------points_local ??????????????? + ??????--------')
    # print(points_local)
    # Group points by block
    blocks = [np.zeros((l, points.shape[1])) for l in block_len]#blocks: l in block len:number of point are assign to the same block
    #points shape[1]=6. (x,y,z and 3 colors)
    blocks_last_idx = np.zeros(len(block_len), dtype=np.uint32)
    # print('---------blocks shape--------')
    # for i in blocks:
    #     print(i.shape)
    # print('---------blocks last idx shape--------')
    # print(blocks_last_idx.shape)
    for i, b_idx in enumerate(block_idx):
        blocks[b_idx][blocks_last_idx[b_idx]] = points_local[i]#store point coordinate in blocks
        blocks_last_idx[b_idx] += 1         #increase counter by 1
    # print('---------blocks--------')
    # print(blocks)
    # print('---------blocks last idx--------')
    # print(blocks_last_idx)
    # print('---------compare to block len--------')
    # print(block_len) 
    # Build binary string recursively using the block_ids
    # after the first time call partition octree fc, the point will be quantize to fit the
    # after this call, number of box and level will be hoa hop
    # print('-------level--------')
    # print(level)
    # print("--------begin get binstr---------")
    _, binstr = partition_octree_rec(block_ids_unique, [0, 0, 0], (2 ** level) * np.array([1, 1, 1]), level)
    # print("--------ending of get binstr---------")

    return blocks, binstr


def departition_octree(blocks, binstr_list, bbox_min, bbox_max, level):
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)

    blocks = blocks.copy()
    binstr_list = binstr_list.copy()
    binstr_idxs = np.zeros(len(binstr_list), dtype=np.uint8)
    children_counts = np.zeros(len(binstr_list), dtype=np.uint32)

    binstr_list_idx = 0
    block_idx = 0
    cur_level = 1

    bbox_stack = [(bbox_min, bbox_max)]
    parents_stack = []

    while block_idx < len(blocks):
        child_found = False
        # Find next child at current level
        while binstr_list[binstr_list_idx] != 0 and not child_found:
            if (binstr_list[binstr_list_idx] & 1) == 1:
                v = binstr_idxs[binstr_list_idx]
                cur_bbox = compute_new_bbox(v, *bbox_stack[-1])
                if cur_level == level:
                    # Leaf node: decode current block
                    blocks[block_idx] = blocks[block_idx] + np.pad(cur_bbox[0], [0, blocks[block_idx].shape[1] - 3])
                    # print(f'Read block {block_idx} at binstr {binstr_list_idx} ({cur_bbox})')
                    block_idx += 1
                else:
                    # print(f'Child found at idx {binstr_idxs[binstr_list_idx]} for binstr {binstr_list_idx}')
                    # Non leaf child: stop reading current binstr
                    child_found = True

            binstr_list[binstr_list_idx] >>= 1
            binstr_idxs[binstr_list_idx] += 1

        if child_found:
            # Child found: descend octree
            bbox_stack.append(cur_bbox)
            parents_stack.append(binstr_list_idx)
            for i in range(len(parents_stack)):
                children_counts[parents_stack[i]] += 1
            # Go to child
            cur_level += 1
            binstr_list_idx += children_counts[parents_stack[-1]]
            # print(f'Descend to {binstr_list_idx}')
        else:
            # No children left: ascend octree
            binstr_list_idx = parents_stack.pop()
            cur_level -= 1
            bbox_stack.pop()
            # print(f'Ascend to {binstr_list_idx}')

    return blocks

