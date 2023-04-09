import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from modules.readout_function import AvgReadout, MaxReadout, MinReadout, WeightedAvgReadout
from modules.evolution_info_getter import EvolutionInfoGetter, AdaptiveFusion
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False, pretrained_emb=None, pretrained_seq=None, seq_len=None, ei_mode=None):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    # embedding可训练形式
    if pretrained_emb == None:
      #self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
      self.node_raw_features = nn.Parameter(torch.from_numpy(node_features).to(torch.float32)).to(device)
    else:
      #self.node_raw_features = pretrained_emb
      self.node_raw_features = nn.Parameter(pretrained_emb, requires_grad=True).to(device)
    
    if edge_features != None:
      #self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
      self.edge_raw_features = nn.Embedding(edge_features.shape[0], edge_features.shape[1]).to(device)
      self.n_edge_features = self.edge_raw_features.shape[1]
    else:
      self.edge_raw_features = None
      self.n_edge_features = 0

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    self.readout = AvgReadout()
    #self.marginloss = nn.MarginRankingLoss(margin=0.3, reduction='mean')
    self.marginloss = nn.TripletMarginLoss(margin=0.3, p=2, reduction='mean')
    
    ###### pretrain_seq ######
    self.pretrained_seq = pretrained_seq
    self.ei_mode = ei_mode
    self.seq_len = seq_len

    self.evolution_info_getter = EvolutionInfoGetter(in_shape=self.embedding_dimension,
                                                     out_shape=self.embedding_dimension, seq_len=self.seq_len,
                                                     mode = self.ei_mode)
    
    self.adaptive_fusion_func =  AdaptiveFusion(info_dim1=self.embedding_dimension, info_dim2=self.embedding_dimension, 
                                                out_dim=self.embedding_dimension)

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    if (self.pretrained_seq != None) and (self.seq_len != 0):
      self.affinity_score = MergeLayer(2*self.n_node_features, 2*self.n_node_features,
                                      2*self.n_node_features, 1)
    else:
      self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                      self.n_node_features, 1)      


  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding
  

  def compute_cl_embeddings(self, source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors=20):
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
        "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]

    return source_node_embedding, destination_node_embedding 


  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()


  def compute_edge_probabilities_with_evolution_info(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
    
    source_node_seq, destination_node_seq, negative_node_seq = self.pretrained_seq[source_nodes], self.pretrained_seq[destination_nodes], self.pretrained_seq[negative_nodes]
    
    if self.ei_mode == 'trans':
      node_seqs = torch.cat([torch.cat([source_node_seq, source_node_embedding.unsqueeze(1)], dim=1), torch.cat([destination_node_seq, destination_node_embedding.unsqueeze(1)], dim=1),
                                          torch.cat([negative_node_seq, negative_node_embedding.unsqueeze(1)], dim=1)], dim=0)
    else:
      node_seqs = torch.cat([source_node_seq, destination_node_seq, negative_node_seq], dim=0)
    
    nodes_ei = self.evolution_info_getter(node_seqs)
    nodes_emb = torch.cat([source_node_embedding, destination_node_embedding, negative_node_embedding], dim=0)
    #print(nodes_ei.shape, nodes_emb.shape, flush=True)
    nodes_ei_fusion = self.adaptive_fusion_func(nodes_ei.squeeze(0), nodes_emb)
    
    source_node_ei_fusion = nodes_ei_fusion[:n_samples]
    destination_node_ei_fusion = nodes_ei_fusion[n_samples:2*n_samples]
    negative_node_ei_fusion = nodes_ei_fusion[2*n_samples:]

    source_node_emb_with_ei = torch.cat([source_node_embedding, source_node_ei_fusion], dim=1)
    destination_node_emb_with_ei = torch.cat([destination_node_embedding, destination_node_ei_fusion], dim=1)
    negative_node_emb_with_ei = torch.cat([negative_node_embedding, negative_node_ei_fusion], dim=1)

    score = self.affinity_score(torch.cat([source_node_emb_with_ei, source_node_emb_with_ei], dim=0),
                                torch.cat([destination_node_emb_with_ei,
                                           negative_node_emb_with_ei], dim=0)).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()


  def embedding_ei_fusion(self, source_nodes, destination_nodes, source_node_embedding, destination_node_embedding):
    n_samples = len(source_nodes)
    source_node_seq, destination_node_seq = self.pretrained_seq[source_nodes], self.pretrained_seq[destination_nodes]
    
    if self.ei_mode == 'trans':
      node_seqs = torch.cat([torch.cat([source_node_seq, source_node_embedding.unsqueeze(1)], dim=1), 
                            torch.cat([destination_node_seq, destination_node_embedding.unsqueeze(1)], dim=1)], dim=0)
    else:
      node_seqs = torch.cat([source_node_seq, destination_node_seq], dim=0)
    
    nodes_ei = self.evolution_info_getter(node_seqs)
    nodes_emb = torch.cat([source_node_embedding, destination_node_embedding], dim=0)
    #print(nodes_ei.shape, nodes_emb.shape, flush=True)
    nodes_ei_fusion = self.adaptive_fusion_func(nodes_ei.squeeze(0), nodes_emb)
    
    source_node_ei_fusion = nodes_ei_fusion[:n_samples]
    destination_node_ei_fusion = nodes_ei_fusion[n_samples:]
    source_node_emb_with_ei = torch.cat([source_node_embedding, source_node_ei_fusion], dim=1)
    destination_node_emb_with_ei = torch.cat([destination_node_embedding, destination_node_ei_fusion], dim=1)
    return source_node_emb_with_ei, destination_node_emb_with_ei


  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update


  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    if self.edge_raw_features != None:
      edge_features = self.edge_raw_features[edge_idxs]
    else:
      edge_features = None

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)
    
    if edge_features != None:
      source_message = torch.cat([source_memory, destination_memory, edge_features, source_time_delta_encoding], dim=1)
    else:
      source_message = torch.cat([source_memory, destination_memory, source_time_delta_encoding], dim=1)
    
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages


  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
    

  def get_full_embedding(self, isSave=True, save_path=None):
    '''
    获取(并存储)所有节点的embedding信息
    '''
    
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

    # Compute the embeddings using the embedding module
    node_embedding = memory
    
    if isSave:
      torch.save(node_embedding, save_path)

    return node_embedding
  
  '''
  def get_all_node_embedding(self, isSave=True, save_path=None):
    node_embedding = self.node_raw_features

    if isSave:
      torch.save(node_embedding, save_path)

    return node_embedding


  def get_all_node_embedding_v2(self, isSave=True, save_path=None):
    node_embedding = None
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update 
      node_embedding = self.node_raw_features + memory
    else:
      node_embedding = self.node_raw_features

    if isSave:
      torch.save(node_embedding, save_path)

    return node_embedding
  '''

  def pretrain(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20, s_cl=True, t_cl=True, k_hop=1, tau=0.8):
    '''
    链接预测+结构对比+时序对比v3
    '''
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding, negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    struct_loss = None
    if s_cl == 1:
      # 结构对比
      comb_l = list(zip(source_nodes, destination_nodes, edge_times))
      random.shuffle(comb_l) # seed
      src_l_shuffle, dst_l_shuffle, ts_l_shuffle = zip(*comb_l)

      pos_embed_batch, neg_embed_batch = np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]]), np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]])
      if k_hop == 1:
        pos_ngh_node_batch_l, pos_ngh_eidx_batch_l, pos_ngh_t_batch_l = self.neighbor_finder.get_temporal_neighbor(source_nodes, edge_times, n_neighbors=n_neighbors)
        neg_ngh_node_batch_l, neg_ngh_eidx_batch_l, neg_ngh_t_batch_l = self.neighbor_finder.get_temporal_neighbor(src_l_shuffle, ts_l_shuffle, n_neighbors=n_neighbors)
      else:
        pos_ngh_node_batch_l, pos_ngh_eidx_batch_l, pos_ngh_t_batch_l = self.neighbor_finder.find_k_hop(k_hop, source_nodes, edge_times, n_neighbors=n_neighbors)
        neg_ngh_node_batch_l, neg_ngh_eidx_batch_l, neg_ngh_t_batch_l = self.neighbor_finder.find_k_hop(k_hop, src_l_shuffle, ts_l_shuffle, n_neighbors=n_neighbors)

      for counter, (pos_idx_l, neg_idx_l) in enumerate(zip(pos_ngh_node_batch_l, neg_ngh_node_batch_l)):
        pos_embed = memory[pos_idx_l, :]
        #pos_embed = self.embedding_module.compute_embedding(memory=memory, source_nodes=pos_idx_l, timestamps=pos_t_l, 
        #                                                    n_layers=self.n_layers, n_neighbors=n_neighbors)
        neg_embed = memory[neg_idx_l, :]
        #neg_embed = self.embedding_module.compute_embedding(memory=memory, source_nodes=neg_idx_l, timestamps=neg_t_l, 
        #                                                    n_layers=self.n_layers, n_neighbors=n_neighbors)
        pos_readout = self.readout(pos_embed).cpu().detach().numpy()
        neg_readout = self.readout(neg_embed).cpu().detach().numpy()
        pos_embed_batch[counter] = pos_readout
        neg_embed_batch[counter] = neg_readout
          
      pos_embed_batch = torch.from_numpy(pos_embed_batch).to(source_node_embedding.device)
      neg_embed_batch = torch.from_numpy(neg_embed_batch).to(source_node_embedding.device)

      #logits_pos = 0.5*F.cosine_similarity(source_node_embedding, pos_embed_batch, dim = -1)+0.5
      #logits_neg = 0.5*F.cosine_similarity(source_node_embedding, neg_embed_batch, dim = -1)+0.5
      #logits_pos = torch.sigmoid(torch.sum(source_node_embedding * pos_embed_batch, dim = -1))
      #logits_neg = torch.sigmoid(torch.sum(source_node_embedding * neg_embed_batch, dim = -1))
      
      #ones = torch.ones(logits_pos.size(0)).to(logits_pos.device)
      #struct_loss = self.marginloss(logits_neg, logits_pos, ones)
      struct_loss = self.marginloss(source_node_embedding, pos_embed_batch, neg_embed_batch)

    temporal_loss = None
    if t_cl == 1:
      # 时序对比
      temp_embed_batch, inv_embed_batch = np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]]), np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]])
      if k_hop == 1:
        temp_ngh_node_batch_l, temp_ngh_eidx_batch_l, temp_ngh_t_batch_l =  self.neighbor_finder.temporal_contrast_sampler(source_nodes, edge_times, num_neighbors=n_neighbors, inv=False, tau=tau)
        inv_ngh_node_batch_l, inv_ngh_eidx_batch_l, inv_ngh_t_batch_l = self.neighbor_finder.temporal_contrast_sampler(source_nodes, edge_times, num_neighbors=n_neighbors, inv=True, tau=tau)
      else:
        temp_ngh_node_batch_l, temp_ngh_eidx_batch_l, temp_ngh_t_batch_l =  self.neighbor_finder.find_k_hop_temporal(k_hop, source_nodes, edge_times, num_neighbors=n_neighbors, inv=False, tau=tau)
        inv_ngh_node_batch_l, inv_ngh_eidx_batch_l, inv_ngh_t_batch_l = self.neighbor_finder.find_k_hop_temporal(k_hop, source_nodes, edge_times, num_neighbors=n_neighbors, inv=True, tau=tau)

      for counter, (temp_idx_l, inv_idx_l) in enumerate(zip(temp_ngh_node_batch_l, inv_ngh_node_batch_l)):
        temp_embed = memory[temp_idx_l, :]
        #temp_embed = self.embedding_module.compute_embedding(memory=memory, source_nodes=temp_idx_l, timestamps=temp_t_l, 
        #                                                    n_layers=self.n_layers, n_neighbors=n_neighbors)
        inv_embed = memory[inv_idx_l, :]
        #inv_embed = self.embedding_module.compute_embedding(memory=memory, source_nodes=inv_idx_l, timestamps=inv_t_l, 
        #                                                    n_layers=self.n_layers, n_neighbors=n_neighbors)
        temp_readout = self.readout(temp_embed).cpu().detach().numpy()
        inv_readout = self.readout(inv_embed).cpu().detach().numpy()
        temp_embed_batch[counter] = temp_readout
        inv_embed_batch[counter] = inv_readout
      
      temp_embed_batch = torch.from_numpy(temp_embed_batch).to(source_node_embedding.device)
      inv_embed_batch = torch.from_numpy(inv_embed_batch).to(source_node_embedding.device)

      #logits_temp = 0.5*F.cosine_similarity(source_node_embedding, temp_embed_batch, dim=-1)+0.5
      #logits_inv = 0.5*F.cosine_similarity(source_node_embedding, inv_embed_batch, dim=-1)+0.5
      #logits_temp = torch.sigmoid(torch.sum(source_node_embedding * temp_embed_batch, dim = -1))
      #logits_inv = torch.sigmoid(torch.sum(source_node_embedding * inv_embed_batch, dim = -1))
      
      #ones = torch.ones(logits_temp.size(0)).to(logits_temp.device)
      #temporal_loss = self.marginloss(logits_inv, logits_temp, ones)      
      temporal_loss = self.marginloss(source_node_embedding, temp_embed_batch, inv_embed_batch)

    return pos_score.sigmoid(), neg_score.sigmoid(), struct_loss, temporal_loss
    

  def pretrain_ts(self, source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors=20):
    '''
    重新设计结构对比和时序对比v2
    '''
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes]) # 2*node_pairs
    #nodes = np.array(source_nodes)
    timestamps = np.concatenate([edge_times, edge_times])
    #timestamps = np.array(edge_times)
    
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update
      
      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)
    
    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)
    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(nodes, self.memory.messages)

        assert torch.allclose(memory[nodes], self.memory.get_memory(nodes), atol=1e-5), \
        "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(nodes)
      
      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
    
    comb_l = list(zip(source_nodes, destination_nodes, edge_times))
    random.shuffle(comb_l)
    src_l_shuffle, dst_l_shuffle, ts_l_shuffle = zip(*comb_l)

    # 结构对比
    pos_embed_batch, neg_embed_batch = np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]]), np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]])
    pos_ngh_node_batch_l, pos_ngh_eidx_batch_l, pos_ngh_t_batch_l = self.neighbor_finder.get_temporal_neighbor(source_nodes, edge_times, n_neighbors=n_neighbors)
    neg_ngh_node_batch_l, neg_ngh_eidx_batch_l, neg_ngh_t_batch_l = self.neighbor_finder.get_temporal_neighbor(src_l_shuffle, ts_l_shuffle, n_neighbors=n_neighbors)
    for counter, (pos_idx_l, neg_idx_l) in enumerate(zip(pos_ngh_node_batch_l, neg_ngh_node_batch_l)):
      pos_embed = memory[pos_idx_l, :]
      neg_embed = memory[neg_idx_l, :]
      pos_readout = self.readout(pos_embed).cpu().detach().numpy()
      neg_readout = self.readout(neg_embed).cpu().detach().numpy()
      pos_embed_batch[counter] = pos_readout
      neg_embed_batch[counter] = neg_readout
    
    pos_embed_batch = torch.from_numpy(pos_embed_batch).to(source_node_embedding.device)
    neg_embed_batch = torch.from_numpy(neg_embed_batch).to(source_node_embedding.device)

    #logits_pos = 0.5*F.cosine_similarity(source_node_embedding, pos_embed_batch, dim = -1)+0.5
    #logits_neg = 0.5*F.cosine_similarity(source_node_embedding, neg_embed_batch, dim = -1)+0.5
    logits_pos = torch.sigmoid(torch.sum(source_node_embedding * pos_embed_batch, dim = -1))
    logits_neg = torch.sigmoid(torch.sum(source_node_embedding * neg_embed_batch, dim = -1))
    
    ones = torch.ones(logits_pos.size(0)).to(logits_pos.device)
    struct_loss = self.marginloss(logits_neg, logits_pos, ones)

    #时序对比
    temp_embed_batch, inv_embed_batch = np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]]), np.empty([source_node_embedding.shape[0], source_node_embedding.shape[1]])
    temp_ngh_node_batch_l, temp_ngh_eidx_batch_l, temp_ngh_t_batch_l =  self.neighbor_finder.temporal_contrast_sampler(source_nodes, edge_times, num_neighbors=n_neighbors, inv=False)
    inv_ngh_node_batch_l, inv_ngh_eidx_batch_l, inv_ngh_t_batch_l = self.neighbor_finder.temporal_contrast_sampler(source_nodes, edge_times, num_neighbors=n_neighbors, inv=True)
    for counter, (temp_idx_l, inv_idx_l) in enumerate(zip(temp_ngh_node_batch_l, inv_ngh_node_batch_l)):
      temp_embed = memory[temp_idx_l,:]
      inv_embed = memory[inv_idx_l,:]
      temp_readout = self.readout(temp_embed).cpu().detach().numpy()
      inv_readout = self.readout(inv_embed).cpu().detach().numpy()
      temp_embed_batch[counter] = temp_readout
      inv_embed_batch[counter] = inv_readout
    
    temp_embed_batch = torch.from_numpy(temp_embed_batch).to(source_node_embedding.device)
    inv_embed_batch = torch.from_numpy(inv_embed_batch).to(source_node_embedding.device)

    #logits_temp = 0.5*F.cosine_similarity(source_node_embedding, temp_embed_batch, dim=-1)+0.5
    #logits_inv = 0.5*F.cosine_similarity(source_node_embedding, inv_embed_batch, dim=-1)+0.5
    logits_temp = torch.sigmoid(torch.sum(source_node_embedding * temp_embed_batch, dim = -1))
    logits_inv = torch.sigmoid(torch.sum(source_node_embedding * inv_embed_batch, dim = -1))
    
    ones = torch.ones(logits_temp.size(0)).to(logits_temp.device)
    temporal_loss = self.marginloss(logits_inv, logits_temp, ones)

    return struct_loss, temporal_loss
