import math
import logging
import time
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
import random

from evaluation.evaluation import eval_edge_prediction, eval_node_classification
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, MLP
from utils.data_processing import get_data, compute_time_statistics, get_data_no_label, get_data_node_classification
from model.mapper import DistributionMLP, DenoiseMLP
from modules.wasserstein_distance import SinkhornDistance
from modules.sliced_wasserstein_distance import sliced_wasserstein_distance
from modules.evolution_info_getter import EvolutionInfoGetter, AdaptiveFusion
#from geomloss import SamplesLoss


### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=256, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') # 0.0001
parser.add_argument('--weight_decay', type=float, default=0.01, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping') #######
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=64, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=64, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=64, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=64, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--pretrained_model_path', type=str, default='./saved_models/tgn-lastfm_pretrain-lpscl-lastfm_pretrain.pth', help='pretrained model path')
parser.add_argument('--pretrained_emb_path', type=str, default='./saved_embed/tgn-lastfm_pretrain-lpscl-lastfm_pretrain-emb.pth', help='pretrained embedding path')
parser.add_argument('--pretrained_seq_path', type=str, default='./saved_seq/tgn-lastfm_pretrain-lpscl-lastfm_pretrain-seq.pth', help='pretrained embedding path')
parser.add_argument('--pretrained', type=int, default=1, help='pretrained/no pretrained')
parser.add_argument('--use_seq', type=int, default=1, help='whether to use sequence')
parser.add_argument('--data_type', type=str, default="gowalla", help='Type of dataset')
parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')
#parser.add_argument('--emb_mode', type=int, default=2, help='1: only node; 2: noly mem; 3: node+mem')
parser.add_argument('--data_path', type=str, default="./", help='path of data')
parser.add_argument('--model_path', type=str, default="./", help='path of model')
parser.add_argument('--log_path', type=str, default="./", help='path of log')
parser.add_argument('--check_path', type=str, default="./", help='path of checkpoints')
parser.add_argument('--emb_path', type=str, default="./", help='path of embedding')
parser.add_argument('--result_path', type=str, default="./", help='path of result')
parser.add_argument('--ei_mode', type=str, default="rnn", help='evolution info getter mode')
parser.add_argument('--use_validation', action='store_true', help='Whether to use a validation set')

############## load the pretrained model and embedding for downstream learning ###############
# wikipedia
# ./saved_models/wiki-pretrain-wikipedia_pretrain.pth
# ./saved_embed/wiki-pretrain-wikipedia_pretrain-emb.pth

# reddit
# ./saved_models/reddit-pretrain-reddit_pretrain.pth
# ./saved_embed/reddit-pretrain-reddit_pretrain-emb.pth

# lastfm
# ./saved_models/tgn-lastfm_pretrain-lastfm_pretrain.pth
# ./saved_embed/tgn-lastfm_pretrain-lastfm_pretrain-emb.pth

# ml1m
# ./saved_models/ml1m-pretrain-ml1m_pretrain.pth
# ./saved_embed/ml1m-pretrain-ml1m_pretrain-emb.pth

# prefix rule:
# {model}-{dataset}
# command rule:
# nohup python downstream.py --use_memory --prefix {model}-{dataset}-(down)/(no) -d {dataset} &>> {dataset}_(down)/(no).out &

############# run command for pretrained model #################
# nohup python downstream.py --use_memory --prefix tgn-lastfm_downstream-down -d lastfm_downstream --pretrained 1 &>> lastfm_down.out &

############# run command for no pretrain model #################
# nohup python downstream.py --use_memory --prefix tgn-lastfm_downstream-no -d lastfm_downstream --pretrained 0 &>> lastfm_no.out &
# nohup python downstream.py --use_memory --prefix tgn-ml10m_downstream-no -d ml10m_downstream --pretrained 0 &>> ml10m_no.out &
# nohup python downstream.py --use_memory --prefix tgn-meituanv2_downstream-no -d meituanv2_downstream --pretrained 0 --gpu 2 --patience 5 &>> ./out/meituanv2/meituanv2_no.out &


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

#Path("./saved_models/").mkdir(parents=True, exist_ok=True)
#Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = args.model_path+f'/saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: args.check_path+f'/saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'
EMB_SAVE_PATH = args.emb_path+f'/saved_embed/{args.prefix}-{args.data}-emb.pth'
PLOT_SAVE_PREFIX = f'./plot/{args.data}'

### set up logger
'''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
'''
print(args, flush=True)


### Extract data for training, validation and testing
full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.use_validation, have_edge=False, data_type=args.data_type, task_type=args.task_type, 
  seed=args.seed, data_path=args.data_path)

max_idx = max(full_data.unique_nodes)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform, args.seed, max_node_idx=max_idx)

'''
# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform, args.seed)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=args.seed)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=args.seed)
'''

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
print("$$$ experimental device: "+device_string, flush=True)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)


def pretrained_model_match(chkpt):
    chkpt.pop('affinity_score.fc1.weight')
    chkpt.pop('affinity_score.fc1.bias')
    chkpt.pop('affinity_score.fc2.weight')
    chkpt.pop('affinity_score.fc2.bias')
    #chkpt.pop('memory_updater.memory_updater.weight_ih')
    #chkpt.pop('memory_updater.memory_updater.weight_hh')
    #chkpt.pop('memory_updater.memory_updater.bias_ih')
    #chkpt.pop('memory_updater.memory_updater.bias_hh')

auc_runs = []

for i in range(args.n_runs):
  if args.n_runs == 1:
    set_seed(args.seed)
  else:
    set_seed(i)
  results_path = args.result_path+"/results/{}_{}.pkl".format(args.prefix, i) if i > 0 else args.result_path+"/results/{}.pkl".format(args.prefix)
  #Path("results/").mkdir(parents=True, exist_ok=True)
  pretrained_emb = None
  pretrained_seqs = None
  seq_length = 0
  if args.pretrained==1:
    # 加载预训练embedding
    print("######loading the pre-trained embedding...######", flush=True)
    #pretrained_emb = torch.nn.Parameter(torch.load(args.pretrained_emb_path, map_location='cpu'), requires_grad=False).to(device)
    pretrained_emb = torch.load(args.pretrained_emb_path, map_location='cpu').to(device)
    print("pretrained emb shape: {}, node feature shape: {}".format(pretrained_emb.shape, node_features.shape), flush=True)

    print("######loading the pre-trained updated sequence...######", flush=True)
    #pretrained_emb = torch.nn.Parameter(torch.load(args.pretrained_emb_path, map_location='cpu'), requires_grad=False).to(device)
    pretrained_seqs = torch.load(args.pretrained_seq_path, map_location='cpu').to(device)
    _, seq_length, _ = pretrained_seqs.shape
    if args.use_seq == 0:
      seq_length = 0
    print("pretrained updated sequence shape: {}".format(pretrained_seqs.shape), flush=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep, pretrained_emb=pretrained_emb, pretrained_seq=pretrained_seqs, seq_len=seq_length, ei_mode=args.ei_mode)
  
  if args.use_seq == 1:
    decoder = MLP(2*args.node_dim, drop=DROP_OUT)
  else:
    decoder = MLP(args.node_dim, drop=DROP_OUT)

  if args.pretrained==1:
    print("######loading the pre-trained model parameters...######", flush=True)
    chkpt = torch.load(args.pretrained_model_path, map_location='cpu')
    pretrained_model_match(chkpt)
    # 加载预训练参数
    tgn.load_state_dict(chkpt, strict=False)
  
  decoder_loss_criterion = torch.nn.BCELoss()
  #if args.pretrained == 1:
  #  optimizer = torch.optim.AdamW([
  #    {'params': tgn.parameters()},
  #    {'params': denoiseMLP.parameters()}
  #  ], lr=LEARNING_RATE, weight_decay=args.weight_decay)
  #else:
  if args.use_seq == 1:
    optimizer = torch.optim.AdamW([{"params":tgn.parameters()}, {"params":decoder.parameters()}], lr=LEARNING_RATE, weight_decay=args.weight_decay)
  else:
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
  #optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)

  tgn = tgn.to(device)
  decoder = decoder.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  print('num of training instances: {}'.format(num_instance), flush=True)
  print('num of batches per epoch: {}'.format(num_batch), flush=True)
  idx_list = np.arange(num_instance)

  epoch_times = []
  total_epoch_times = []
  train_losses = []
  val_aucs = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    #tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    print('start {} epoch'.format(epoch), flush=True)
    loss = 0
    for k in range(num_batch):
      # Custom loop to allow to perform backpropagation only every a certain number of batches
      batch_idx = k

      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], train_data.destinations[start_idx:end_idx]
      edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
      timestamps_batch = train_data.timestamps[start_idx:end_idx]
      edge_idxs_batch = full_data.edge_idxs[start_idx: end_idx]
      labels_batch = train_data.labels[start_idx: end_idx]

      size = len(sources_batch)
      if args.use_seq:
        tgn = tgn.train()
      else:
        tgn = tgn.eval()
      decoder = decoder.train()

      optimizer.zero_grad()
      with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                     destinations_batch,
                                                                                     destinations_batch,
                                                                                     timestamps_batch,
                                                                                     edge_idxs_batch,
                                                                                     NUM_NEIGHBORS)
      if args.use_seq == 1:
        source_emb_ei, destination_emb_ei = tgn.embedding_ei_fusion(source_nodes=sources_batch, destination_nodes=destinations_batch, 
                                          source_node_embedding=source_embedding, destination_node_embedding=destination_embedding)
      else:
        source_emb_ei = source_embedding
        destination_emb_ei = destination_embedding

      labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
      pred = decoder(source_emb_ei).sigmoid()
      decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      decoder_loss.backward()
      optimizer.step()
      loss = loss + decoder_loss.item()
    
    train_losses.append(loss / num_batch)
    val_auc = eval_node_classification(tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS, use_seq=args.use_seq)
    val_aucs.append(val_auc)
    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    pickle.dump({
      "val_aps": val_aucs,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
    }, open(results_path, "wb"))

    print(f'Epoch {epoch}: train loss: {loss / num_batch}, val auc: {val_auc}, time: {epoch_time}', flush=True)
    if args.use_validation:
      if early_stopper.early_stop_check(val_auc):
        print('No improvement over {} epochs, stop training'.format(early_stopper.max_round), flush=True)
        break
      else:
        torch.save(decoder.state_dict(), get_checkpoint_path(epoch))
    
  if args.use_validation:
    print(f'Loading the best model at epoch {early_stopper.best_epoch}', flush=True)
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    decoder.load_state_dict(torch.load(best_model_path))
    print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference', flush=True)
    decoder.eval()
    test_auc = eval_node_classification(tgn, decoder, test_data, full_data.edge_idxs, BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS, use_seq=args.use_seq)
  else:
    # If we are not using a validation set, the test performance is just the performance computed
    # in the last epoch
    test_auc = val_aucs[-1]
  
  auc_runs.append(test_auc)
  pickle.dump({
    "val_aps": val_aucs,
    "test_ap": test_auc,
    "train_losses": train_losses,
  }, open(results_path, "wb"))
  print(f'Run {i}: test auc: {test_auc}', flush=True)

auc_mean, auc_std = np.mean(np.array(auc_runs)), np.std(np.array(auc_runs))
print("************** All runs done! **************", flush=True)
print(f'test auc: {auc_mean} ± {auc_std}', flush=True)
print("********************************************", flush=True)
