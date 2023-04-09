import pandas as pd
import numpy as np
import pickle
import time
import datetime
import logging

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

'''
time transfer:
2014年之前的数据作为预训练数据
2014年及之后作为下游任务数据

field transfer:
ACS类别做预训练数据
Beauty, Luxury和Fashion类别做下游任务数据

time+field transfer:
2014年之前的ACS类别做预训练数据
2014年及之后的Beauty, Luxury和Fashion类别做下游任务数据
'''

#note: 需要统计全局最大与全局最小user&item; 先分类再做remap
classes = ['beauty', 'fashion', 'luxury', 'acs']
split_time = 20170101
logger.info("split timestamp: %d"%(split_time))


# 对id进行重排
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(1,len(key)+1)))
    df.loc[:,col_name] = df[col_name].map(lambda x: m[x])
    return m, key


# 统计dataframe各字段的信息
def data_stat(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(1,len(key)+1)))
    return m

def dataset_analyse(df):
    max_node_id = max(df.u.max(), df.i.max())
    min_node_id = min(df.u.min(), df.i.min())
    max_edge_id = df.idx.max()
    min_edge_id = df.idx.min()
    return max_node_id, min_node_id, max_edge_id, min_edge_id

def reindex(df):
    new_df = df.copy()
    upper_u = df.u.max()
    new_i = df.i + upper_u
    new_df.i = new_i
    #new_df.idx += 1
    return new_df


def dataset_split(df):
    df_pretrain = df[df['datetime']<split_time]
    df_downstream = df[df['datetime']>=split_time]
    return df_pretrain, df_downstream


def df_normalize():
    df = pd.read_csv('./amazon_process.csv', header=0)
    df['idx'] = [i for i in range(df.shape[0])]
    df.rename(columns={'user_id':'u'},inplace=True)
    df.rename(columns={'item_id':'i'},inplace=True)
    user_map, _ = build_map(df, 'u')
    item_map, _ = build_map(df, 'i')
    _, _ = build_map(df, 'idx')
    user_count, item_count, example_count = len(user_map), len(item_map), df.shape[0]
    logger.info("###########################")
    logger.info('Total\tuser_count: %d\titem_count: %d\texample_count: %d'%(user_count, item_count, example_count))
    reindex_df = reindex(df)
    max_node_id, min_node_id, max_edge_id, min_edge_id = dataset_analyse(reindex_df)
    logger.info("max node id: %d\tmax edge id: %d"%(max_node_id, max_edge_id))
    
    # for field-trans
    # field_transfer_data(reindex_df, max_node_id, max_edge_id)
    
    # for time-trans
    df_pretrain, df_downstream = dataset_split(reindex_df)
    df_pretrain = df_pretrain.sort_values('timestamp')
    df_pretrain = df_pretrain.reset_index(drop=True)
    df_downstream = df_downstream.sort_values('timestamp')
    df_downstream = df_downstream.reset_index(drop=True)
    # time transfer for pretrain:
    time_transfer_data(df_pretrain, 'pretrain', max_node_id, max_edge_id)
    # time transfer for downstream:
    time_transfer_data(df_downstream, 'downstream', max_node_id, max_edge_id)



######## for time transfer #########
def time_transfer_data(df, split, max_node_id, max_edge_id):
    for cla in classes:
        now_df = df[df['cate']==cla]
        u_list = pd.to_numeric(now_df['u'])
        i_list = pd.to_numeric(now_df['i'])
        ts_list = pd.to_numeric(now_df['timestamp'])
        idx_list = [i for i in range(len(u_list))]
        result_df = pd.DataFrame({'u': u_list, 'i':i_list, 'ts':ts_list, 'idx':idx_list})

        OUT_DF = './time_trans/ml_amazon_{}_{}.csv'.format(cla, split)
        #OUT_FEAT = './time_trans/ml_gowalla_{}_{}.npy'.format(cla, split)
        OUT_NODE_FEAT = './time_trans/ml_amazon_{}_{}_node.npy'.format(cla, split)
        user_map = data_stat(result_df, 'u')
        item_map = data_stat(result_df, 'i')
        user_count, item_count, example_count = len(user_map), len(item_map), result_df.shape[0]
        node_feat = np.random.uniform(size=(max_node_id + 1, 64))
        #edge_feat = np.random.uniform(size=(max_edge_id + 1, 64))
        result_df.to_csv(OUT_DF)
        #np.save(OUT_FEAT, edge_feat)
        np.save(OUT_NODE_FEAT, node_feat)
        logger.info('%s %s data: user_count: %d\titem_count: %d\texample_count: %d\tprocessed done!'%(cla, split, user_count, item_count, example_count))

        

######## for field transfer ########
def field_transfer_data(df, max_node_id, max_edge_id):
    for cla in classes:
        now_df = df[df['cate']==cla]
        u_list = pd.to_numeric(now_df['u'])
        i_list = pd.to_numeric(now_df['i'])
        ts_list = pd.to_numeric(now_df['timestamp'])
        idx_list = [i for i in range(len(u_list))]
        result_df = pd.DataFrame({'u': u_list, 'i':i_list, 'ts':ts_list, 'idx':idx_list})

        OUT_DF = './field_trans/ml_amazon_{}.csv'.format(cla)
        #OUT_FEAT = './field_trans/ml_gowalla_{}.npy'.format(cla)
        OUT_NODE_FEAT = './field_trans/ml_amazon_{}_node.npy'.format(cla)
        user_map = data_stat(result_df, 'u')
        item_map = data_stat(result_df, 'i')
        user_count, item_count, example_count = len(user_map), len(item_map), result_df.shape[0]
        node_feat = np.random.uniform(size=(max_node_id + 1, 64))
        #edge_feat = np.random.uniform(size=(max_edge_id + 1, 64))
        result_df.to_csv(OUT_DF)
        #np.save(OUT_FEAT, edge_feat)
        np.save(OUT_NODE_FEAT, node_feat)
        logger.info('%s data: user_count: %d\titem_count: %d\texample_count: %d\tprocessed done!'%(cla, user_count, item_count, example_count))



######## for time+field transfer ########
def time_field_transfer_data():
    pass


df_normalize()
