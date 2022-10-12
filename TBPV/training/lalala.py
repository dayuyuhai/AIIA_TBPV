import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import os
import sys
import time
from torchvision import datasets, transforms
import argparse
from torchsummary import summary
from pyntcloud import PyntCloud
from glob import glob
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.training_tools import save_ckp, load_ckp, compute_metric, Rotation, Random_sampling
import datetime
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        """
        
        dots = dots + mask
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        """
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)      
        return x

class ViT(nn.Module):
    def __init__(self, pc_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dim_head, dropout, emb_dropout):
        super().__init__()

        pc_height, pc_width,  pc_length = pair(pc_size)
        patch_height, patch_width, patch_length = (1, 1, 1)

        print("pc length,height, width:", pc_length, pc_height, pc_width)
        print("patch length, height, width: ", patch_length, patch_height, patch_width)

        patch_dim = channels * patch_height * patch_width * patch_length # 1
        num_patches = (pc_height // patch_height) * (pc_width //
                                                patch_width) * (pc_length // patch_length)
        print("num of patches:", num_patches)

        # 分辨率
        self.res = pc_height 

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (l p3) -> b (h w l) (p1 p2 p3 c)',
                      p1=patch_height, p2=patch_width,  p3=patch_length),
            nn.Linear(patch_dim, dim),
        )

        # concact坐标

        self.pos_embedding = PositionalEncoding(ninp, dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim * 2)  
        )

    def forward(self, pc):
        x = self.to_patch_embedding(pc)
        b, n, _ = x.shape
        print("x shape:", x.shape)    # (b, n, dim)
        # x += self.pos_embedding[:, :(n)]  # before: n + 1
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = x.view(x.shape[0], 2, self.res, self.res, self.res)
        return x                        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PCdataset(Dataset):
    def __init__(self, files, transforms=None):
        # 数据转换成 ndarray 形式
        self.files = np.asarray(files)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        pc = PyntCloud.from_file(self.files[idx])
        points = pc.points.to_numpy()[:, :3]

        if (self.transforms):
            points = self.transforms(points)
        try:  # np.unique 去除重复并排序
            points = np.unique(points, axis=0)
        except:
            return None
        points = torch.from_numpy(points).type(torch.LongTensor)

        print('------------point shape---------------')
        print(points)
        print(points.shape)

        v = torch.ones(points.shape[0])

        # 64 * 64 * 64 的block
        dense_block = torch.sparse.FloatTensor(torch.transpose(
            points, 0, 1), v, torch.Size([64, 64, 64])).to_dense().view(1, 64, 64, 64)

        print(dense_block.shape, torch.max(dense_block), torch.min(dense_block), torch.count_nonzero(dense_block))

        return dense_block

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def data_collector(training_dirs, params):
    total_files = []
    for training_dir in training_dirs:
        training_dir = training_dir + '**/*.ply'

        # 四个文件夹，依次找到每个文件夹下的所有 .ply 文件
        files = glob(training_dir, recursive=True)

        # 打印当前文件夹下的 .ply 文件个数
        print('Total files: ', len(files))

        total_files_len = len(files)

        # total_files中是所有 .ply 文件
        total_files = np.concatenate((total_files, files), axis=0)
        print('Selected ', len(files), ' from ',
              total_files_len, ' in ', training_dir)

    # assert：如果它的条件返回错误，则终止程序执行
    assert len(total_files) > 0

    '''
        random.shuffle()用于将一个列表中的元素打乱顺序
        这个方法不会生成新的列表，只是将原列表的次序打乱
    '''
    rn.shuffle(total_files)

    # .ply 文件的个数
    print('Total blocks for training: ', len(total_files))

    # os.path.split() : 以 "PATH" 中最后一个 '/' 作为分隔符，分隔后，将索引为0的视为目录（路径），将索引为1的视为文件名
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1]
                         for x in total_files])

    # 所有训练数据 .ply
    files_train = total_files[files_cat == 'train']

    # 所有验证数据 .ply
    files_valid = total_files[files_cat == 'test']

    # rotation = Rotation(64)
    # sampling = Random_sampling()
    # transforms_ = Compose([rotation, sampling])
    # ,transforms.ToTensor()

    training_set = PCdataset(files_train)

    # 用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。
    '''
    params:
        batchsize：每个 batch 加载多少个样本
        shuffle：用来打乱数据集中数据顺序，避免数据投入的顺序对网络造成影响，True：每个 epoch 都重新排序
        num_workers：dataloading 所用到的子进程数
    collate_fn：设置batch数据拼接方式
    '''

    training_generator = torch.utils.data.DataLoader(
        training_set, collate_fn=collate_fn, **params)

    # Validation data
    valid_set = PCdataset(files_valid)
    valid_generator = torch.utils.data.DataLoader(
        valid_set, collate_fn=collate_fn, **params)

    return training_generator, valid_generator


def train(use_cuda, batch_size, low_res, max_epochs, group, output_model, dataset_path, valid_loss_min, model,
          optimizer, start_epoch, ds_level):

    # 日志
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = output_model + 'log' + current_time + '/train'
    test_log_dir = output_model + 'log' + current_time + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    # 设置检查点  pt:torch 的一种序列化文件
    checkpoint_path = output_model + "current_checkpoint.pt"
    best_model_path = output_model + "best_model.pt"

    # 用来处理浮点数误差
    eps = 1e-8

    # batchsize：每个 batch 加载多少个样本
    # shuffle：用来打乱数据集中数据顺序，避免数据投入的顺序对网络造成影响，True：每个 epoch 都重新排序
    # num_workers：dataloading 所用到的子进程数
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    # 将模型加载到 gpu
    device = torch.device("cuda" if args.usecuda else "cpu")
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])
    model = model.to(device)

    # data_collector
    training_generator, valid_generator = data_collector(dataset_path, params)

    # 对 64 * 64 * 64 大小的 block，如果处理除第一组以外的点，就不做处理
    # 对 32 * 32 * 32 大小的 block，如果处理除第一组以外的点，就下采样一次

    maxpool_n1 = nn.Sequential(
        *[nn.MaxPool3d(kernel_size=2) for _ in range(ds_level - 1)]
    )

    # 对 64 * 64 * 64 大小的 block，如果处理第一组点，就下采样一次得到 32 * 32 * 32 的 low level 来预测
    maxpool_n = nn.Sequential(
        *[nn.MaxPool3d(kernel_size=2) for _ in range(ds_level)]
    )

    # 训练损失
    train_loss = 0
    train_losses = []
    if start_epoch == 0:
        best_val_epoch = None
    else :
        best_val_epoch = start_epoch - 1
    output_period = len(training_generator) // 2
    block_size = int(low_res * 2)
    idx = np.unravel_index(group, (2, 2, 2))
    print('Start the training for group: ', group,
          ' with lower resoltuion is ', low_res,  ' with batch_size is ', batch_size)

    # 训练
    for epoch in range(start_epoch, max_epochs):
        for batch_idx, x in enumerate(training_generator):
            x = x.to(device)
            # 下采样后大小的块块，即祖先
            input = maxpool_n(x)

            if (epoch == 0 and batch_idx == 0):
                print('Input shape: ', input.shape)

            target = maxpool_n1(x.clone().detach())[:, :, idx[0]:block_size:2, idx[1]:block_size:2, idx[2]:block_size:2].view(x.shape[0], low_res, low_res,
                                                                                                     low_res).type(torch.LongTensor)
            target = target.to(device)
            predict = model(input) + eps

            loss = F.cross_entropy(predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = train_loss + \
                ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

            tp, fp, tn, fn, precision, recall, accuracy, specificity, f1 = compute_metric(predict, target,
                                                                                          train_summary_writer, len(
                                                                                              training_generator) * epoch + batch_idx)

            train_summary_writer.add_scalar(
                "bc/loss", train_loss, len(training_generator) * epoch + batch_idx)

            print("Batch {} over {}:  \tloss : {:.6f}\t accuracy : {:.3f} ".format(
                batch_idx, len(training_generator), train_loss, accuracy), end='\r')
            if (batch_idx % output_period == 0):
                print("Batch {} over {}:  \tloss : {:.6f}\t accuracy : {:.3f} tp : {:.2f} fp : {:.2f} tn : {:.2f} fn : {:.2f} f1 : {:.4f}".format(
                    batch_idx, len(training_generator), train_loss, accuracy, tp, fp, tn, fn, f1), end='\r')

        train_losses.append(train_loss)
        print("train loss:", train_losses)

        valid_loss = 0

        # 测试时添加model.eval()，保证dropout不会丢神经元，BN 层的均值和方差不会变
        model.eval()

        # 验证
        for batch_idx, x in enumerate(valid_generator):
            x = x.to(device)
            # x:(32, 1, 64, 64, 64)
            input = maxpool_n(x)

            target = maxpool_n1(x.clone().detach())[:, :, idx[0]:block_size:2, idx[1]:block_size:2, idx[2]:block_size:2].view(
                x.shape[0], low_res, low_res,
                low_res).type(torch.LongTensor)

            target = target.to(device)

            output = model(input) + eps

            loss = F.cross_entropy(output, target)

            valid_loss = valid_loss + \
                ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            valid_loss = valid_loss
            del loss, target, output

        test_summary_writer.add_scalar("bc/loss", valid_loss, epoch)

        print('Training for group: ', group, ' downsample level: ', ds_level)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss))

        # saving model
        # create checkpoint variable and add important data

        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # 如果没存档过检查点，或当前验证损失小于之前的最小验证损失，就保存当前最好的模型
        if valid_loss <= valid_loss_min or best_val_epoch == None:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
            best_val_epoch = epoch

        # 训练十次也不如之前好了，就提前终止
        if (epoch-best_val_epoch >= 10):
            print('Early stopping detected')
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-scratch", '--scratch', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=False,
                        help='Training from scratch or checkpoint')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True,
                        help='using cuda or not')
    parser.add_argument("-batch", '--batch', type=int,
                        default=8,
                        help='batch size')

    parser.add_argument("-dim", '--dim', type=int,
                        default=1024,
                        help='linear output dim')
    parser.add_argument("-depth", '--depth', type=int,
                        default=3,
                        help='num of transformer')
    parser.add_argument("-mlp_dim", '--mlp_dim', type=int,
                        default=256,
                        help='num of hidden layer neural')
    parser.add_argument("-heads", '--heads', type=int,
                        default=8,
                        help='num of heads')

    parser.add_argument("-blocksize", '--blocksize', type=int,
                        default=64,
                        help='training block size')
    parser.add_argument("-epoch", '--epoch', type=int,
                        default=10,
                        help='number of epochs')
    parser.add_argument("-downsample", '--dslevel', type=int,
                        default=2,
                        help='number of downsampling step until group 1 level')

    parser.add_argument("-inputmodel", '--savedmodel',
                        type=str, help='path to saved model file')
    # parser.add_argument("-loss", '--loss_img_name', type=str, help='name of loss image')
    parser.add_argument("-outputmodel", '--saving_model_path',
                        type=str, help='path to output model file')
    parser.add_argument("-dataset", '--dataset',
                        action='append', type=str, help='path to dataset ')
    parser.add_argument("-validation", '--validation',
                        type=str, help='path to validation dataset ')
    parser.add_argument("-portion_data", '--portion_data', type=float,
                        default=1,
                        help='portion of dataset to put in training, densier pc are selected first')
    args = parser.parse_args()
    low_res = int(64 / (2 ** args.dslevel))
    model = ViT(low_res, low_res, 2, args.dim, args.depth, args.heads, args.mlp_dim, 'cls', 1, 64, 0.1, 0.1)
    # 截止的最小损失
    valid_loss_min = np.Inf
    device = torch.device("cuda" if args.usecuda else "cpu")
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=[4, 5, 6, 7])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    start_epoch = 0

    output_path = args.saving_model_path + 'G' + \
        str(args.group) + '_lres' + str(low_res) + '/'
    os.makedirs(output_path, exist_ok=True)

    # 从头训练
    if (args.scratch):
        print("Training from scratch \n")
        train(args.usecuda, args.batch, low_res, args.epoch, args.group, output_path, args.dataset,
              valid_loss_min, model, optimizer, start_epoch, args.dslevel)

    # 从检查点训练：进度保存 具体解释参考 https://zhuanlan.zhihu.com/p/410548507
    else:
        ckp_path = output_path + "current_checkpoint.pt"
        model, optimizer, start_epoch, valid_loss_min = load_ckp(
            ckp_path, model, optimizer)
        print('Successfully loaded model \n')
        train(args.usecuda, args.batch, low_res, args.epoch, args.group, output_path, args.dataset,
              valid_loss_min, model, optimizer, start_epoch, args.dslevel)
