import argparse
import numpy
from distance_utils import *

parser = argparse.ArgumentParser()
# =========== 数据文件参数 ===========
parser.add_argument('--data_path', type=str, default='./data', help="the path of data file")
parser.add_argument('--data_log', type=str, default='CP_exp_n64_e128_c5000_t10', help="data file")
parser.add_argument('--logs_path', type=str, default='./logs', help="the path of logs file")
parser.add_argument('--z_log', type=str, default='exp_CP_exp_n64_e128_c5000_t10_2021-12-30T01%24%14%866753')
parser.add_argument('--occ_threshold', type=float, default=0.01)
parser.add_argument('--threshold', type=float, default=0.0019)
parser.add_argument('--thre_down', type=float, default=0)
parser.add_argument('--thre_up', type=float, default=0.005)
parser.add_argument('--step', type=float, default=0.0001)
parser.add_argument('--distance', type=str, default='WS',
                    help="ED, MD, Cos, KL, JS, WS")
parser.add_argument('--plot', action='store_true', default=False,
                    help="plot the metrics")
parser.add_argument('--undirected', action='store_true', default=False,
                    help="plot the metrics")
parser.add_argument('--filter', action='store_true', default=False,
                    help="是否通过频率筛选")
parser.add_argument('--analysis', action='store_true', default=False,
                    help="是否画图分析")

args = parser.parse_args()

net, z, occ, undirected_net = load_net_Z(args)
# 筛选共现率较高的节点对
node_pair_index = find_possible_node_pair(occ, net, args.occ_threshold, args.filter, args.analysis)
if args.undirected and args.distance != "KL":  # 是否采用无向图 KL散度不需要无向图
    net = undirected_net
if args.distance == "ED":
    dis = ED(net.shape[0], z, node_pair_index)
elif args.distance == "MD":
    dis = MD(net.shape[0], z, node_pair_index)
elif args.distance == "cos":
    dis = cosine_dis(net.shape[0], z, node_pair_index)
elif args.distance == "KL":
    dis = KL(net.shape[0], z, node_pair_index)
elif args.distance == "JS":
    dis = JS(net.shape[0], z, node_pair_index)
elif args.distance == "WS":
    dis = WS(net.shape[0], z, node_pair_index)
else:
    print("WARNING: please input the correct distance!")

# 画直方图
if args.analysis:
    plot_distribution(dis)

if args.plot:
    plot_metrics(dis, args.thre_down, args.thre_up, args.step, net, args.distance)
else:
    if args.distance == "cos":
        pred_net = prediction_cos(dis, args.threshold)
    else:
        pred_net = prediction(dis, args.threshold)

    metrics(pred_net, net, is_print=True)
