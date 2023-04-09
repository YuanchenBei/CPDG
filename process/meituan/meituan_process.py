import pandas as pd
import numpy as np


def mt_process():        
    u_list, i_list, ts_list = [], [], []
    noclick = 0
    with open('./meituan_train.csv') as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            isclick = int(e[2])
            if isclick != 1:
                noclick += 1
            ts = int(e[5])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
    
    if noclick != 0:
        print("have unclick data in training set!")
    else:
        print("checked in training set")
    df1 = pd.DataFrame({'u': u_list, 'i':i_list, 'ts':ts_list})
    
    ###########
    u_list, i_list, ts_list = [], [], []
    noclick = 0
    with open('./meituan_val.csv') as f:
        s = next(f)
        print(s)
        noclick = 0
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            isclick = int(e[2])
            if isclick != 1:
                noclick += 1
            ts = int(e[5])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
    
    if noclick != 0:
        print("have unclick data in val set!")
    else:
        print("checked in val set")
    df2 = pd.DataFrame({'u': u_list, 'i':i_list, 'ts':ts_list})

    ############
    u_list, i_list, ts_list = [], [], []
    noclick = 0
    with open('./meituan_test.csv') as f:
        s = next(f)
        print(s)
        noclick = 0
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            isclick = int(e[2])
            if isclick != 1:
                noclick += 1
            ts = int(e[5])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
    
    if noclick != 0:
        print("have unclick data in test set!")
    else:
        print("checked in test set")
    df3 = pd.DataFrame({'u': u_list, 'i':i_list, 'ts':ts_list})

    df = pd.concat([df1, df2, df3])
    print(df.head())
    df.sort_values('ts', inplace=True)
    print(df.head())
    df.to_csv('./meituanv2.csv', header=True, index=False)


def time_split():
    reviews_df  = pd.read_csv('./meituanv2.csv', header=0)
    print(reviews_df.head())
    split_time = reviews_df.ts.quantile([0.60, 0.70, 0.80])
    #df_pretrain = sorted_df[sorted_df['timestamp'] < 19064806.0]
    #df_downstream = sorted_df[sorted_df['timestamp'] >= 19064806.0]
    return split_time


def reindex(df):
    new_df = df.copy()
    upper_u = df.u.max()
    new_i = df.i + upper_u
    new_df.i = new_i
    #new_df.idx += 1
    return new_df


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(1,len(key)+1)))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


def preprocess_pretrain(data_name):
    u_list, i_list, ts_list = [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = int(e[2])
            
            feat = np.random.uniform(size=64)

            # 拉出前60%的数据作为预训练数据
            if ts < 1646896000:
                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                idx_list.append(idx)
                feat_l.append(feat)

    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list,
                         'idx':idx_list}), np.array(feat_l)


def preprocess_downstream(data_name):
    u_list, i_list, ts_list = [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = int(e[2])
            
            feat = np.random.uniform(size=64)

            # 拉出后40%的数据作为下游任务数据
            if ts >= 1646896000:
                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                idx_list.append(idx)
                feat_l.append(feat)

    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list,
                         'idx':idx_list}), np.array(feat_l)


def run():
    data_name = 'meituanv2'
    PATH = './{}.csv'.format(data_name)
    data_name_pre = data_name+'_pretrain'
    data_name_down = data_name+'_downstream'
    # pre-train
    OUT_DF_pre = './ml_{}.csv'.format(data_name_pre)
    OUT_FEAT_pre = './ml_{}.npy'.format(data_name_pre)
    OUT_NODE_FEAT_pre = './ml_{}_node.npy'.format(data_name_pre)
    # downstream
    OUT_DF_down = './ml_{}.csv'.format(data_name_down)
    OUT_FEAT_down = './ml_{}.npy'.format(data_name_down)
    OUT_NODE_FEAT_down = './ml_{}_node.npy'.format(data_name_down)

    # [0.6, 0.7, 0.8]: [92140173.4, 103764807.2, 115147271.8]
    # print(time_select(PATH))

    df_pre, feat_pre = preprocess_pretrain(PATH)
    user_map, user_key = build_map(df_pre, 'u')
    item_map, item_key = build_map(df_pre, 'i')
    edge_map, edge_key = build_map(df_pre, 'idx')
    user_count, item_count, example_count = len(user_map), len(item_map), df_pre.shape[0]
    print('Pretrain Data: user_count: %d\titem_count: %d\texample_count: %d'%(user_count, item_count, example_count))

    new_df_pre = reindex(df_pre)
    empty_pre = np.zeros(feat_pre.shape[1])[np.newaxis, :]
    feat_pre = np.vstack([empty_pre, feat_pre])

    max_idx_pre = max(new_df_pre.u.max(), new_df_pre.i.max())
    rand_feat_pre = np.random.uniform(size=(max_idx_pre + 1, feat_pre.shape[1]))
    #rand_feat_pre = np.zeros((max_idx_pre+1, feat_pre.shape[1]))

    print(feat_pre.shape, rand_feat_pre.shape)

    new_df_pre.to_csv(OUT_DF_pre)
    np.save(OUT_FEAT_pre, feat_pre)
    np.save(OUT_NODE_FEAT_pre, rand_feat_pre)

    #######################
    df_down, feat_down = preprocess_downstream(PATH)
    user_map, user_key = build_map(df_down, 'u')
    item_map, item_key = build_map(df_down, 'i')
    edge_map, edge_key = build_map(df_down, 'idx')

    user_count, item_count, example_count = len(user_map), len(item_map), df_down.shape[0]
    print('Downstream Data: user_count: %d\titem_count: %d\texample_count: %d'%(user_count, item_count, example_count))

    new_df_down = reindex(df_down)
    empty_down = np.zeros(feat_down.shape[1])[np.newaxis, :]
    feat_down = np.vstack([empty_down, feat_down])

    max_idx_down = max(new_df_down.u.max(), new_df_down.i.max())
    rand_feat_down = np.random.uniform(size=(max_idx_down+1, feat_down.shape[1]))

    print(feat_down.shape, rand_feat_down.shape)

    new_df_down.to_csv(OUT_DF_down)
    np.save(OUT_FEAT_down, feat_down)
    np.save(OUT_NODE_FEAT_down, rand_feat_down)


#mt_process()
#run()
ti_list = time_split()
print(ti_list)

'''
0.6    1.646896e+09  1646896000
0.7    1.647223e+09  1647223000
0.8    1.647673e+09  1647673000
'''
