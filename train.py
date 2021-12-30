import os
import time
import argparse
import datetime
import pickle
import numpy as np
import torch
from torch import optim
from utils import *
from modules import *

parser = argparse.ArgumentParser()
# =========== 数据文件参数 ===========
parser.add_argument('--data_path', type=str, default='./data', help="the path of data file")
parser.add_argument('--data_log', type=str, default='CP_exp_n64_e128_c5000_t10', help="data file")
parser.add_argument('--save_folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')

# ===========数据集参数 ===========
parser.add_argument("--num_nodes", type=int, default=64)
parser.add_argument("--num_cascade", type=int, default=64)

# =========== 训练参数 ===========
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')  # 显式调用时，不用cuda
parser.add_argument('--step', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Number of samples per batch.')
parser.add_argument('--beta', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.001)

# =========== 模型参数 ===========
parser.add_argument('--encoder_hid1', type=list, default=[48, 32])
parser.add_argument('--encoder_out', type=int, default=16)
parser.add_argument('--decoder_hid1', type=list, default=[32, 48])

args = parser.parse_args()
args.node_feat = args.num_nodes
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 固定随机性
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    timestamp = timestamp.replace(':', '%')
    timestamp = timestamp.replace('.', '%')
    save_folder = '{}/exp_{}_{}/'.format(args.save_folder, args.data_log, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided to save log and model!")

data_loader, occ = load_data(args)  # data_loader([node_id, cascade_num, feature_dim], node_id) 相当于data第一个维度为样本维度

encoder = Encoder(args.node_feat, args.encoder_hid1, args.encoder_out, device, args.num_cascade)
decoder = Decoder(args.encoder_out, args.decoder_hid1, args.node_feat, device, args.num_cascade)
if args.cuda:
    encoder.cuda()
    decoder.cuda()
    occ = occ.cuda()

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)


def train(epoch, best_loss):
    t = time.time()
    loss_train = []
    encoder.train()
    decoder.train()
    for batch_idx, (X, node_id) in enumerate(data_loader):
        if args.cuda:
            X, node_id = X.cuda(), node_id.cuda()
        optimizer.zero_grad()

        z = encoder(X)
        x_hat = decoder(z)

        loss = loss_function(X, x_hat, args.beta)
        loss.backward()
        optimizer.step()

        loss_train.append(loss.item())
        if args.step:
            print("Epoch:{:04}".format(epoch),
                  'loss_train: {:.10f}'.format(loss.item()))
    print("Epoch:{:04}".format(epoch),
          'loss_train: {:.10f}'.format(np.mean(loss_train)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(loss_train) < best_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print("Epoch:{:04}".format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(loss_train)


# train model
t_total = time.time()
best_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    epoch_loss = train(epoch, best_loss)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

if log is not None:
    print(save_folder)
    log.close()
