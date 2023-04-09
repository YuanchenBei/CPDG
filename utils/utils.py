import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, seed=0, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform, seed=seed)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.
    获取边的时序为cut time之前的邻居
    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.
    采样时序邻居
    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
    return neighbors, edge_idxs, edge_times
  
  def find_k_hop(self, k, src_idx_l, cut_time_l, n_neighbors=20):
    """
    Sampling the k-hop sub graph
    """
    x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, n_neighbors)
    node_records = [x]
    eidx_records = [y]
    t_records = [z]
    for _ in range(k -1):
      ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
      orig_shape = ngn_node_est.shape
      ngn_node_est = ngn_node_est.flatten()
      ngn_t_est = ngh_t_est.flatten()
      out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, n_neighbors)
      #logger.info("orig_shape: {}, out_ngh_node_batch shape: {}".format(orig_shape, out_ngh_node_batch.shape))
      #out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, n_neighbors) # [N, *([num_neighbors] * k)]
      out_ngh_node_batch = out_ngh_node_batch.reshape(orig_shape[0], orig_shape[1]*n_neighbors)
      out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(orig_shape[0], orig_shape[1]*n_neighbors)
      out_ngh_t_batch = out_ngh_t_batch.reshape(orig_shape[0], orig_shape[1]*n_neighbors)
      node_records.append(out_ngh_node_batch)
      eidx_records.append(out_ngh_eidx_batch)
      t_records.append(out_ngh_t_batch)
      #logger.info("orig_shape: {}, out_ngh_node_batch shape new: {}".format(orig_shape, out_ngh_node_batch.shape))
    node_records = np.concatenate([node_re for node_re in node_records], axis=-1)
    eidx_records = np.concatenate([eidx_re for eidx_re in eidx_records], axis=-1)
    t_records = np.concatenate([t_re for t_re in t_records], axis=-1)
    #logger.info("orig_shape: {}, node_batch shape new new: {}".format(orig_shape, node_records.shape))
    return node_records, eidx_records, t_records


  def get_temporal_probability(self, ngh_ts, inv=False, tau=0.7):
    '''
    按照时间修改采样概率
    '''
    tp = np.array(ngh_ts)
    if inv:
        tp = np.max(tp)-tp
    #tp = tp - np.max(tp, axis=0, keepdims=True)
    #tp = np.exp(tp)/np.sum(np.exp(tp), axis=0, keepdims=True)
    max_t = np.max(tp, axis=0)
    min_t = np.min(tp, axis=0)
    tp = (tp-min_t)/(max_t-min_t) if max_t>min_t else (tp-min_t)
    tp = np.exp(tp/tau)/np.sum(np.exp(tp/tau), axis=0, keepdims=True)
    #print(tp)
    return tp


  def temporal_contrast_sampler(self, src_idx_l, cut_time_l, num_neighbors=20, inv=False, tau=0.7):
    '''
    时序对比的采样器
    '''
    assert(len(src_idx_l) == len(cut_time_l))
    
    out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
    out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
    out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
    
    for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
      ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

      if len(ngh_idx) > 0:
        temporal_probability = self.get_temporal_probability(ngh_ts, inv=inv)
        sampled_idx = np.random.choice(np.arange(0, len(ngh_idx)), size=num_neighbors, p=temporal_probability)
        out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
        out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
        out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
            
        # resort based on time
        pos = out_ngh_t_batch[i, :].argsort()
        out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
        out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
        out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
    
    return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch
  

  def find_k_hop_temporal(self, k, src_idx_l, cut_time_l, num_neighbors=20, inv=False, tau=0.7):
    """
    Sampling the k-hop sub graph in temporal probability
    """
    x, y, z = self.temporal_contrast_sampler(src_idx_l, cut_time_l, num_neighbors, inv)
    node_records = [x]
    eidx_records = [y]
    t_records = [z]
    for _ in range(k -1):
      ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
      orig_shape = ngn_node_est.shape
      ngn_node_est = ngn_node_est.flatten()
      ngn_t_est = ngh_t_est.flatten()
      out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.temporal_contrast_sampler(ngn_node_est, ngn_t_est, num_neighbors, inv, tau)
      out_ngh_node_batch = out_ngh_node_batch.reshape(orig_shape[0], orig_shape[1]*num_neighbors) # [N, *([num_neighbors] * k)]
      out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(orig_shape[0], orig_shape[1]*num_neighbors)
      out_ngh_t_batch = out_ngh_t_batch.reshape(orig_shape[0], orig_shape[1]*num_neighbors)
      node_records.append(out_ngh_node_batch)
      eidx_records.append(out_ngh_eidx_batch)
      t_records.append(out_ngh_t_batch)
    node_records = np.concatenate([node_re for node_re in node_records], axis=-1)
    eidx_records = np.concatenate([eidx_re for eidx_re in eidx_records], axis=-1)
    t_records = np.concatenate([t_re for t_re in t_records], axis=-1)
    return node_records, eidx_records, t_records
