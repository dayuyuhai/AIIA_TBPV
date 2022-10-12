import logging
from socket import PF_CAN
from time import perf_counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
from os.path import join, basename, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
import pandas as pd
import argparse
import functools
from tqdm import tqdm
from multiprocessing import Pool
from utils.octree_partition import partition_octree


def arr_to_pc(arr, cols, types):
    d = {}
    for i in range(arr.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = arr[:, i].astype(dtype)
    print('--------d--------------')
    print(d)
    df = pd.DataFrame(data=d)
    print('--------df--------------')
    print(df)
    pc = PyntCloud(df)
    print('--------pc--------------')
    print(pc)
    return pc


def process(path, args):
    ori_path = join(args.source, path)
    # print("--------oripath--------")
    # print(ori_path)
    target_path, _ = splitext(join(args.dest, path))
    # print("--------target_path--------")
    # print(target_path)
    target_folder, _ = split(target_path)
    # print("--------target_folder--------")
    # print(target_folder)
    makedirs(target_folder, exist_ok=True)

    pc = PyntCloud.from_file(ori_path)
    # print('-------pc---------')
    # print(pc)
    points = pc.points.values
    # print('-----------pcvalue---------')
    # print(points)
    # print('-------------------pc col, pc type-----------')
    # print(pc.points.columns)
    # print(pc.points.dtypes)
    bbox_min = [0, 0, 0]
    bbox_max = [args.vg_size, args.vg_size, args.vg_size]
    # print('-------------------partition_begin-------------------')
    blocks, _ = partition_octree(points, bbox_min, bbox_max, args.level)
    # print('--------------------partition_end---------------------')
    # print('----------block at end---------')
    # print(blocks)

    for i, block in enumerate(blocks):
        final_target_path = target_path + f'_{i:03d}{args.target_extension}'
        # print('------finalpath---------')
        # print(final_target_path)
        logger.debug(f"Writing PC {ori_path} to {final_target_path}")
        cur_pc = arr_to_pc(block, pc.points.columns, pc.points.dtypes)
        cur_pc.to_file(final_target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_pc_octree_blocks.py',
        description='Converts a folder containing meshes to point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--vg_size', type=int, help='Voxel Grid resolution for x, y, z dimensions', default=64)
    parser.add_argument('--level', type=int, help='Octree decomposition level.', default=3)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    assert args.vg_size > 0, f'vg_size must be positive'

    # datasets/MPEG/10bitdepth/文件夹下所有点云
    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source):] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    '''
        1. functools.partial(func, args)
            预先给函数提供参数并且返回一个对象
            https://blog.csdn.net/cassiePython/article/details/76653897?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-76653897-blog-80769967.topnsimilarv1&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-76653897-blog-80769967.topnsimilarv1&utm_relevant_index=1
        2. imap(function, iter)
            它将一次遍历可迭代对象中的一个元素，并将它们分别发送给工作进程。
            第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
        3. tqdm(可迭代对象)
            在使用Python处理比较耗时操作的时候, 需要通过进度条将处理情况进行可视化展示
            装饰一个可迭代对象，返回一个与原始可迭代对象完全相同的迭代器
            每次请求值时都会打印一个动态更新的进度条。
            https://blog.csdn.net/weixin_44878336/article/details/124894210
    '''
    # with Pool() as p:
    #     process_f = functools.partial(process, args=args)
    #     list(tqdm(p.imap(process_f, files), total=files_len)) # 这里list的作用没懂
    #     # Without parallelism
    #     # list(tqdm((process_f(f) for f in files), total=files_len))
    process_f = functools.partial(process, args=args)
    list(tqdm((process_f(f) for f in files), total=files_len))
    logger.info(f'{files_len} models written to {args.dest}')


# python -m utils.ds_pc_octree_blocks datasettest/MPEG/10bitdepth/ datasettest/MPEG/10bitdepth_2_oct4/ --vg_size 1024 --level 4