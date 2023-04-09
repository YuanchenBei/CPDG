import pandas as pd
import numpy as np

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            
            
            
            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    print(idx_list[-1])
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)


def reindex(df):
    assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    
    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())
    
    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    
    print(new_df.u.max())
    print(new_df.i.max())
    
    return new_df


def run(data_name):
    PATH = './{}.csv'.format(data_name)
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    
    df, feat = preprocess(PATH)
    new_df = reindex(df)
    
    pretrain_time = list(np.quantile(new_df.ts, [0.60]))[0]
    print(pretrain_time)
    df_pretrain = new_df[new_df.ts < pretrain_time]
    df_downstream = new_df[new_df.ts >= pretrain_time]

    pretrain_size = df_pretrain.shape[0]
    
    print(feat.shape)
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat]) # 给0号index的边加入全零向量
    
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    
    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


def run_splited(data_name):
    PATH = './{}.csv'.format(data_name)

    OUT_DF_pretrain = './processed/ml_{}_pretrain.csv'.format(data_name)
    OUT_DF_downstream = './processed/ml_{}_downstream.csv'.format(data_name)

    OUT_FEAT = './processed/ml_{}.npy'.format(data_name)

    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df)
    
    pretrain_time = list(np.quantile(new_df.ts, [0.60]))[0]
    print("pre-train timestamp before: ", pretrain_time)
    df_pretrain = new_df[new_df.ts < pretrain_time]
    df_pretrain.reset_index(inplace=True)
    df_downstream = new_df[new_df.ts >= pretrain_time]
    df_downstream.reset_index(inplace=True)

    pretrain_size = df_pretrain.shape[0]
    downstream_size = df_downstream.shape[0]
    print(pretrain_size, downstream_size)
    
    print(feat.shape)
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat]) # 给0号index的边加入全零向量
    
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    
    print(feat.shape)
    df_pretrain.to_csv(OUT_DF_pretrain)
    df_downstream.to_csv(OUT_DF_downstream)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

run_splited('reddit')
