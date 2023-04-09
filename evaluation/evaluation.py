import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, is_pretrained=0, use_seq=0, batch_size=1000):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  val_f1_micro, val_f1_macro = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      if is_pretrained == 1 and use_seq == 1:
        pos_prob, neg_prob = model.compute_edge_probabilities_with_evolution_info(sources_batch, destinations_batch,
                                                              negative_samples, timestamps_batch,
                                                              edge_idxs_batch, n_neighbors)
      else:
        pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                              negative_samples, timestamps_batch,
                                                              edge_idxs_batch, n_neighbors)
      
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_f1_micro.append(f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='micro'))
      val_f1_macro.append(f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='macro'))

  return np.mean(val_ap), np.mean(val_auc), np.mean(val_f1_micro), np.mean(val_f1_macro)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors, use_seq=0):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      if use_seq == 1:
        source_emb_ei, destination_emb_ei = tgn.embedding_ei_fusion(source_nodes=sources_batch, destination_nodes=destinations_batch, 
                                          source_node_embedding=source_embedding, destination_node_embedding=destination_embedding)
      else:
        source_emb_ei = source_embedding
        destination_emb_ei = destination_embedding

      pred_prob_batch = decoder(source_emb_ei).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc


'''
对比学习损失在验证集上的计算值
'''
def eval_contrastive_learning(model, data, n_neighbors, alpha=0.5, batch_size=200):
  val_loss = []
  with torch.no_grad():
    model = model.eval()

    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      size = len(sources_batch)

      struct_loss, temporal_loss = tgan.pre_train(sources_batch, destinations_batch, timestamps_batch, n_neighbors)
      now_loss = ALPHA*struct_loss + (1-ALPHA)*temporal_loss
      val_loss.append(now_loss)

  return np.mean(val_loss)
