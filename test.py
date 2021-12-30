import pickle
import argparse
import os
import torch
from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, default='./logs/exp_CP_exp_n64_e128_c5000_t10_2021-12-29T03%30%05%948686')
args1 = parser.parse_args()

f = open(os.path.join(args1.log_path, 'metadata.pkl'), 'rb')
args = pickle.load(f)['args']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_file = os.path.join(args1.log_path, 'encoder.pt')
decoder_file = os.path.join(args1.log_path, 'decoder.pt')
data_loader, occ = load_data(args)

# 固定随机性
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

encoder = Encoder(args.node_feat, args.encoder_hid1, args.encoder_out, device, args.num_cascade)
decoder = Decoder(args.encoder_out, args.decoder_hid1, args.node_feat, device, args.num_cascade)

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    occ = occ.cuda()


def test():
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    Z = torch.zeros((args.num_nodes, args.encoder_out)).to(device)
    X_hat = torch.zeros((args.num_nodes, args.num_cascade, args.num_nodes)).to(device)
    encoder.eval()
    decoder.eval()
    for batch_idx, (X, node_id) in enumerate(data_loader):
        if args.cuda:
            X, node_id = X.cuda(), node_id.cuda()

        z = encoder(X)
        Z[node_id] = z
        x_hat = decoder(z)
        X_hat[node_id] = x_hat

    np.save(os.path.join(args1.log_path, 'Z.npy'), Z.cpu().detach().numpy())
    np.save(os.path.join(args1.log_path, 'X_hat.npy'), X_hat.cpu().detach().numpy())


test()
