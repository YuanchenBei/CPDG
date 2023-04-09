import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import gzip
from collections import defaultdict
import pickle
from datetime import datetime

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def get_data():
    df = getDF(f'./acs/Arts_Crafts_and_Sewing.json.gz')
    print("finish stage 1")
    data = df[['reviewerID', 'asin', "reviewTime", 'unixReviewTime']]
    print("finish stage 2")
    data.columns = ['user_id','item_id', "datetime", "timestamp"]
    print("finish stage 3")
    data.to_csv('./acs/acs.csv', index=False)
    print("finish stage 4! all done!")

def process_time(time_str):
    #re_str = time_str[7:]+time_str[0:2]+time_str[3:5]
    temp = datetime.strptime(time_str, "%m %d, %Y")
    return temp.strftime("%Y%m%d")

def preprocess_data(name_list):
    u_map = {}
    i_map = {}

    u_id = 0
    i_id = 0

    u_list = []
    i_list = []
    ts_list = []
    dt_list = []
    cate_list = []

    #user_count = defaultdict(int)
    #item_count = defaultdict(int)
    for data_name in name_list:
        data = pd.read_csv(f"./{data_name}/{data_name}.csv", header=0)

        for idx, row in data.iterrows():
            u, i, dt, ts = row['user_id'], row['item_id'], row['datetime'], row['timestamp']
            if u not in u_map:
                u_map[u] = u_id
                u_id += 1
            if i not in i_map:
                i_map[i] = i_id
                i_id += 1
            u_list.append(u_map[u])
            i_list.append(i_map[i])
            dt_list.append(process_time(dt))
            ts_list.append(float(ts))
            cate_list.append(data_name)
        print("%s processed done!"%data_name)

    csv_data = pd.DataFrame({'user_id': u_list, 'item_id': i_list, 'datetime':dt_list, 'timestamp': ts_list, 'cate': cate_list})
    csv_data.to_csv(f'./amazon_process.csv', index=False)

name_list = ['beauty', 'fashion', 'luxury', 'acs']
preprocess_data(name_list)
