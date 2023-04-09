import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_data_no_label


### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=256, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') #0.0001
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
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
parser.add_argument('--alpha', type=float, default=0.5, help='balance parameter between structure and temporal')
parser.add_argument('--s_cl', type=int, default=1, help='whether to use structure contrast')
parser.add_argument('--t_cl', type=int, default=1, help='whether to use temporal contrast')
parser.add_argument('--k_hop', type=int, default=2, help='hops in the sampled subgraph')
parser.add_argument('--data_type', type=str, default="gowalla", help='Type of dataset')
parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')
parser.add_argument('--seed', type=int, default=0, help='random seed for all')
parser.add_argument('--tau', type=float, default=0.7, help='temperature parameter in temporal subgraph sampling')
#parser.add_argument('--emb_mode', type=int, default=2, help='1: only node; 2: noly mem; 3: node+mem')
parser.add_argument('--data_path', type=str, default="./", help='path of data')
parser.add_argument('--model_path', type=str, default="./", help='path of model')
parser.add_argument('--log_path', type=str, default="./", help='path of log')
parser.add_argument('--check_path', type=str, default="./", help='path of checkpoints')
parser.add_argument('--emb_path', type=str, default="./", help='path of embedding')
parser.add_argument('--seq_path', type=str, default="./", help='path of pretrain seq')
parser.add_argument('--result_path', type=str, default="./", help='path of result')
parser.add_argument('--seq_len', type=int, default=10, help='random seed for all')

############# run command for pretraining #################
# prefix rule:
# {model}-{dataset}
# command rule:
# nohup python pretrain_cl.py --use_memory --prefix {model}-{dataset} -d {dataset} &>> {dataset}_pre.out &

# tgn
# nohup python pretrain_cl.py --use_memory --prefix tgn-lastfm_pretrain -d lastfm_pretrain &>> lastfm_pre.out &

# jodie
# nohup python pretrain_cl.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie-lastfm_pretrain -d lastfm_pretrain &>> lastfm_pre.out &

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

def get_pretrain_seq(source_seqs, current_seq=None, first_content=False):
  if first_content:
    return source_seqs.unsqueeze(1)
  else:
    return torch.cat([source_seqs, current_seq.unsqueeze(1)], dim=1)

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
ALPHA = args.alpha

#Path(f"/saved_models").mkdir(parents=True, exist_ok=True)
#Path(f"/saved_checkpoints").mkdir(parents=True, exist_ok=True)
#Path(f"/saved_embed").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = args.model_path+f'/saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: args.check_path+f'/saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'
EMB_SAVE_PATH = args.emb_path+f'/saved_embed/{args.prefix}-{args.data}-emb.pth'
SEQ_SAVE_PATH = args.seq_path+f'/saved_seq/{args.prefix}-{args.data}-seq.pth'

### set up logger
'''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(args.log_path+'/log/{}.log'.format(str(time.time())))
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
# 数据划分对齐
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data_no_label(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, \
                              have_edge=False, data_type=args.data_type, task_type=args.task_type, seed=args.seed, data_path=args.data_path)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform, args.seed)

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

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
print("$$$ experimental device: "+device_string, flush=True)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  set_seed(i)
  results_path = args.result_path+"/results/{}_{}.pkl".format(args.prefix, i) if i > 0 else args.result_path+"/results/{}.pkl".format(args.prefix)
  #Path("/results/").mkdir(parents=True, exist_ok=True)

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
            dyrep=args.dyrep)
  
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.AdamW(tgn.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  seq_update_step = num_batch // args.seq_len

  print('num of training instances: {}'.format(num_instance), flush=True)
  print('num of batches per epoch: {}'.format(num_batch), flush=True)
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  #early_stopper = EarlyStopMonitor(max_round=args.patience)
  best_val = 0.0
  best_epoch = 0
  best_seqs = None
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []
    first_seq = True
    pretrain_seqs = None

    print('start {} epoch'.format(epoch), flush=True)
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      batch_idx = k

      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], train_data.destinations[start_idx:end_idx]
      edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
      timestamps_batch = train_data.timestamps[start_idx:end_idx]

      size = len(sources_batch)
      _, negatives_batch = train_rand_sampler.sample(size)

      with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)

      tgn = tgn.train()
      #pretrain_mapper = pretrain_mapper.train()
      #downstream_mapper = downstream_mapper.train()
      
      pos_prob, neg_prob, struct_loss, temporal_loss = tgn.pretrain(sources_batch, destinations_batch, negatives_batch,
                                                                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, args.s_cl, 
                                                                    args.t_cl, args.k_hop, args.tau)

      loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      if args.s_cl == 1 and args.t_cl == 1:
        loss += ALPHA*struct_loss + (1-ALPHA)*temporal_loss
      elif args.s_cl == 1 and args.t_cl != 1:
        loss += struct_loss
      elif args.s_cl != 1 and args.t_cl == 1:
        loss += temporal_loss
    
      if k % 30 == 0:
        print('Total loss: {}, structure loss: {}, temporal loss: {}'.format(loss, struct_loss, temporal_loss), flush=True)

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      ######### save pretrain seq ###########
      if (batch_idx+1)%seq_update_step == 0:
        print('   saving and updating the pretrained sequence...',flush=True)
        now_node_emb = tgn.get_full_embedding(isSave=False)
        if first_seq:
          pretrain_seqs = get_pretrain_seq(now_node_emb, first_content=True)
          first_seq=False
        else:
          pretrain_seqs = get_pretrain_seq(pretrain_seqs, now_node_emb, first_content=False)
        print('   saved the pretrained sequence <= batch {}'.format(batch_idx), flush=True)

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    
    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()


    val_ap, val_auc, val_f1_micro, val_f1_macro = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler, 
                                                                        data=val_data, n_neighbors=NUM_NEIGHBORS)

    
    if USE_MEMORY:
      # val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc, nn_val_f1_micro, nn_val_f1_macro = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler,
                                                data=new_node_val_data, n_neighbors=NUM_NEIGHBORS)

    
    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(train_memory_backup)
    
    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    print('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time), flush=True)
    print('Epoch mean loss: {}'.format(np.mean(m_loss)), flush=True)
    print(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc), flush=True)
    print(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap), flush=True)
    print(
      'val micro-f1: {}, new node val micro-f1: {}'.format(val_f1_micro, nn_val_f1_micro), flush=True)
    print(
      'val macro-f1: {}, new node val macro-f1: {}'.format(val_f1_macro, nn_val_f1_macro), flush=True)
    
    ######### save pretrain seq ###########
    #print('saving and updating the pretrained sequence...',flush=True)
    #now_node_emb = tgn.get_full_embedding(isSave=False)
    #if epoch==0:
    #  pretrain_seqs = get_pretrain_seq(now_node_emb, first_content=True)
    #else:
    #  pretrain_seqs = get_pretrain_seq(pretrain_seqs, now_node_emb, first_content=False)
    #print('saved the pretrained sequence <= epoch {}'.format(epoch), flush=True)

    # Early stopping
    '''
    if early_stopper.early_stop_check(val_auc):
      print('No improvement over {} epochs, stop training'.format(early_stopper.max_round), flush=True)
      print(f'Loading the best model at epoch {early_stopper.best_epoch}', flush=True)
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference', flush=True)
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
    '''
    if val_auc > best_val:
      print('Improved! Saving the best epoch...', flush=True)
      best_val = val_auc
      best_epoch = epoch
      best_seqs = pretrain_seqs
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))


  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  '''
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()
  '''
  print(f'Loading the best model at epoch {best_epoch}, with the best val AUC: {best_val}', flush=True)
  best_model_path = get_checkpoint_path(best_epoch)
  tgn.load_state_dict(torch.load(best_model_path))
  print(f'Loaded the best model at epoch {best_epoch} for inference', flush=True)
  tgn.eval()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_f1_micro, test_f1_macro = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  
  if USE_MEMORY:
    tgn.memory.restore_memory(train_memory_backup)
  

  # Test on unseen nodes
  nn_test_ap, nn_test_auc, nn_test_f1_micro, nn_test_f1_macro = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  print(
    'Test statistics: Old nodes -- auc: {}, ap: {}, micro-f1: {}, macro-f1: {}'.format(test_auc, test_ap, test_f1_micro, test_f1_macro), flush=True)
  print(
    'Test statistics: New nodes -- auc: {}, ap: {}, micro-f1: {}, macro-f1: {}'.format(nn_test_auc, nn_test_ap, nn_test_f1_micro, nn_test_f1_macro), flush=True)
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  print('Saving TGN model', flush=True)
  
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(train_memory_backup)

  # data, isSave=True, save_path=None
  '''
  if args.emb_mode == 1:
    node_emb = tgn.get_all_node_embedding(isSave=True, save_path=EMB_SAVE_PATH)
  elif args.emb_mode == 2:
    node_emb = tgn.get_full_embedding(isSave=True, save_path=EMB_SAVE_PATH)
  else:
    node_emb = tgn.get_all_node_embedding_v2(isSave=True, save_path=EMB_SAVE_PATH)
  '''
  node_emb = tgn.get_full_embedding(isSave=True, save_path=EMB_SAVE_PATH)
  print('Pre-trained node embedding shape: {}, node feature shape: {}'.format(node_emb.shape, node_features.shape), flush=True)
  torch.save(best_seqs, SEQ_SAVE_PATH)
  print('Pre-trained temporal evolution sequence saved.', flush=True)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  print('Pre-trained TGN model and embedding saved.', flush=True)
