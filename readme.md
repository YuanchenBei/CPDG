# CPDG: A contrastive pre-training method for dynamic graph neural networks


### Dependencies
- python >= 3.7
- pandas==1.1.0
- torch==1.6.0
- scikit-learn==0.23.1


### Data Preprocessing
The preprocessing code of all the experimental datasets (Link prediction data: Amazon, Gowalla, Meituan; Node classification data: Wikipedia, MOOC, Reddit) has been place in the **/process** folder.

The source data can be download as below links:

Amazon: https://jmcauley.ucsd.edu/data/amazon

Gowalla: http://www.yongliu.org/datasets.html

Wikipedia: http://snap.stanford.edu/jodie/wikipedia.csv

MOOC: http://snap.stanford.edu/jodie/mooc.csv

Reddit: http://snap.stanford.edu/jodie/reddit.csv


### Running
(i) Determine the data path after placing the preprocessed data.

(ii) Model pre-training through pretrain_cl.py [the example is as follows, find the location of the data through the corresponding path parameter]

`
python3 pretrain_cl.py 
--use_memory
--prefix tgn-amazon_beauty_pretrain_t-ts-bs1024
-d amazon_beauty_pretrain
--bs 1024
--gpu 0
--s_cl 1
--t_cl 1
--alpha 0.5
--seq_len 10
--data_type amazon
--task_type time_trans
--data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--log_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--check_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--emb_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--result_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--seq_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
`

(iii) Perform downstream fine-tuning tasks through downstream.py (link prediction) or downstream_nc.py (node classification) [the example is as follows, find the location of the data through the corresponding path parameter]

`
python3 downstream.py
--use_memory
--prefix tgn-amazon_beauty_downstream_t-lp-bs1024-lr0001
-d amazon_beauty_downstream
--bs 1024
--lr 0.0001
--pretrained 1
--use_seq 1
--data_type amazon
--task_type time_trans
--gpu 0
--n_runs 5
--pretrained_model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run/saved_models/tgn-amazon_beauty_pretrain_t-lp-amazon_beauty_pretrain.pth
--pretrained_emb_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run/saved_embed/tgn-amazon_beauty_pretrain_t-lp-amazon_beauty_pretrain-emb.pth
--pretrained_seq_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run/saved_seq/tgn-amazon_beauty_pretrain_t-lp-amazon_beauty_pretrain-seq.pth
--data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--log_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--check_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--emb_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run
--result_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ycbei/research/run/model-run 
`
