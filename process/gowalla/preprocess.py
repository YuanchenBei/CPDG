import json
import numpy as np
import requests
from jsonsearch import JsonSearch
import pickle
import pandas as pd
import time
import datetime

# 生成时间戳信息
def time_process():
    df = pd.read_csv('./gowalla_checkins.csv', header=0)
    print(df.head())
    print(df.min())
    print(df.max())
    df['timestamp'] = df['datetime'].apply(get_timestamp)
    sorted_df = df.sort_values('datetime')
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df.to_csv('./gowalla_processed.csv',index=False)
    print(sorted_df.head())


def get_timestamp(timeStr):
    #timestamp = datetime.datetime.strptime(timeStr, "%Y-%m-%dT%H:%M:%SZ")
    #timestamp = time.mktime(timestamp)
    timestamp = int(float(time.mktime(time.strptime(timeStr, "%Y-%m-%dT%H:%M:%SZ"))))
    return timestamp


def class_map():
    df = pd.read_csv('./gowalla_spots_subset1.csv', header=0)
    df = df[['id', 'spot_categories']]
    print(df.head())
    df['spot_categories'] = df['spot_categories'].apply(get_cate)
    df.rename(columns={'spot_categories':'cate'},inplace=True)
    df.to_csv('./gowalla_spot_cate.csv',index=False)
    print(df.head())

classes = ['Community', 'Entertainment', 'Food', 'Nightlife', 'Outdoors', 'Shopping', 'Travel']
    
with open('cate_dict.pkl', 'rb') as f:
    cate_dict = pickle.load(f)

def get_cate(cateStr):
    cate = eval(cateStr)[0]['name']
    for cla in classes:
        if cate in cate_dict[cla]:
            cate = cla
            break
    return cate

def build_gowalla_cate():
    df1 = pd.read_csv('./gowalla_processed.csv', header=0)
    df2 = pd.read_csv('./gowalla_spot_cate.csv', header=0)
    df2.rename(columns={'id':'placeid'},inplace=True)
    df = pd.merge(df1, df2, on='placeid')
    print(df.head())
    sorted_df = df.sort_values('datetime')
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df.to_csv('./gowalla_cate_processed.csv',index=False)
    print(sorted_df.head())

def item2class():
    item_class_dict = {}
    classes = ['Community', 'Entertainment', 'Food', 'Nightlife', 'Outdoors', 'Shopping', 'Travel']
    
    with open('cate_dict.pkl', 'rb') as f:
        cate_dict = pickle.load(f)

    for cla in classes:
        item_class_dict.update({cla:[]})

    with open('./gowalla_spot_cate.csv') as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            item_id = int(e[0])
            item_cate = str(e[1])
            for cla in classes:
                if item_cate in cate_dict[cla]:
                    item_class_dict[cla].append(item_id)
                    break
    
    with open('item2class_dict.pkl', 'wb') as f:
        pickle.dump(item_class_dict, f)
  
    return item_class_dict

#time_process()
build_gowalla_cate()

def extract_element_from_json(obj, path):
    def extract(obj, path, ind, arr):
        key = path[ind]
        if ind + 1 < len(path):
            if isinstance(obj, dict):
                if key in obj.keys():
                    extract(obj.get(key), path, ind + 1, arr)
                else:
                    arr.append(None)
            elif isinstance(obj, list):
                if not obj:
                    arr.append(None)
                else:
                    for item in obj:
                        extract(item, path, ind, arr)
            else:
                arr.append(None)
        if ind + 1 == len(path):
            if isinstance(obj, list):
                if not obj:
                    arr.append(None)
                else:
                    for item in obj:
                        arr.append(item.get(key, None))
            elif isinstance(obj, dict):
                arr.append(obj.get(key, None))
            else:
                arr.append(None)
        return arr
    if isinstance(obj, dict):
        return extract(obj, path, 0, [])
    elif isinstance(obj, list):
        outer_arr = []
        for item in obj:
            outer_arr.append(extract(item, path, 0, []))
        return outer_arr


def main(path):
    #response = requests.request("GET", path)
    #data = response.json()
    data = json.load(open(path,'r', encoding='utf-8'))
    #cate_list = extract_element_from_json(data, ["spot_categories", "name"])
    #cate_list = [members.get('name') for members in data.get('spot_categories') ]
    #print(cate_list)
    jsondata = JsonSearch(object=data, mode='j')
    cate_list = jsondata.search_all_value(key='name')
    print(cate_list)

# path = "./gowalla_category_structure.json"
# main(path)

# ['Community', 'Entertainment', 'Food', 'Nightlife', 'Outdoors', 'Shopping', 'Travel']
# ['Community', 'Campus Spot', 'Administration', 'Campus Commons', 'Campus - Other', 'Dormitory', 'Frat House', 'Hall', 'Lab', 'Rec Center', 'Sorority House', 'Student Center', 'Government', 'City Hall', 'Courthouse', 'Fire Station', 'Police Station', 'Post Office', 'Home', 'Apartment', 'Condo', 'Craftsman', 'Duplex', 'Modern', 'Victorian', 'Library', 'Office', 'Corporate Office', 'Coworking', 'Non-Profit', 'Skyscraper', 'Startup', 'Warehouse & Industrial', 'Place of Worship', 'Church', 'Contemporary Church', 'Historic Church', 'Mission', 'Mosque', 'Other - Place of Worship', 'Synagogue', 'Temple', 'School', 'College', 'High School', 'Law School', 'Medical School', 'School - Other', 'Trade/Tech School', 'University', 'Entertainment', 'Aquarium', 'Arcade', 'Arena or Stadium', 'Arena', 'Baseball Stadium', 'Soccer Stadium', 'Stadium', 'Art', 'Asian Art Museum', 'Gallery', 'Modern Art Museum', 'Sculpture', 'Traditional Art Museum', 'Art & Culture', 'Bowling', 'Casino', 'Convention Center', 'Disc Golf Course', 'Golf Course', 'Ice Skating', 'Live Music', 'Local Sports', 'Baseball Field', 'Basketball Court', 'Football Field', 'Soccer Field', 'Tennis Court', 'Movie Theatre', 'Museum', 'History Museum', 'Planetarium', 'Science Museum', 'Other - Entertainment', 'Performing Arts', 'Racetrack', 'Skatepark', 'Theatre', 'Theme Park', 'Zoo', 'Aviary', 'Bears', 'Elephants', 'Giraffes', 'Hippos', 'Lions', 'Other - Zoo', 'Primates', 'Reptiles', 'Tigers', 'Zebras', 'Food', 'African', 'American', 'Asian', 'Asian Food', 'Pan Pacific', 'Sushi', 'Thai', 'Vietnamese', 'Bakery', 'BBQ', 'Breakfast', 'Burgers', 'Coffee Shop', 'Dessert', 'Candy Store', 'Chocolate', 'Frozen Yogurt', 'Ice Cream', 'Snow Cones', 'Diner', 'Doughnuts', 'Fine Dining', 'Fish & Chips', 'French', 'Indian', 'Italian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Other - Food ', 'Pizza', 'Sandwich Shop', 'Seafood', 'Soul Food', 'South American/Latin', 'Steakhouse', 'Street Fare', 'Hot Dogs & Sausages', 'Other - Street Fare', 'Snow Cones', 'Tacos', 'Tea Room', 'Vegetarian', 'Vineyard', 'Wings', 'Nightlife', 'Bar', 'Dancefloor', 'Dive Bar', 'Karaoke', 'Microbrewery', 'Other - Nightlife ', 'Pub', 'Saloon', 'Sports Bar', 'Ultra-lounge', 'Wine Bar', 'Outdoors', 'Architecture', 'Castle', 'Historic Landmark', 'Lighthouse', 'Monument', 'Other - Architecture', 'Tower', 'Beach', 'Canal', 'Cave', 'Cemetery', 'Dog Park', 'Farm', 'Fountain', 'Garden', 'Lake/River', 'Park', 'Campground', 'City Park', 'National Park', 'Park - Other', 'Regional/State Park', 'Trailhead', 'Pavilion', 'Playground', 'Plaza/Square', 'Pool', 'Scenic Lookout', 'Ski & Snowboard Area', 'Chalet', 'Glade', 'Gondola', 'Halfpipe', 'Lifthouse', 'Summit', 'Terrain Park', 'Vineyard', 'Shopping', 'Antiques', 'Apparel', 'Accessories', "Children's Apparel", "Men's Apparel", 'Shoes', "Women's Apparel", 'Bank & Financial', 'Bookstore', 'Crafts & Creative', 'Department Store', 'Discount Store', 'Drugstore & Pharmacy', 'Farmers Market', 'Flower Shop', 'Furniture', 'Gas & Automotive', 'Grocery', 'Gym', 'Hardware Store', 'Health & Fitness', 'Mall', 'Medical', 'Dentist', "Doctor's Office", 'Drugstore & Pharmacy', 'Hospital', 'Other - Medical', 'Movie Rental', 'Music', 'Other - Services', 'Other - Shopping', 'Paper Goods', 'Pet Store', 'Record Store', 'Salon & Barbershop', 'Sports & Outdoors', 'Bike Shop', 'Camping & Outdoors', 'Golf Shop', 'Other Sports', 'Running', 'Skate Shop', 'Ski Shop', 'Tattoo Parlor', 'Technology', 'Thrift Store', 'Tobacco & Cigars', 'Toys & Games', 'Veterinarian', 'Wine & Spirits', 'Travel', 'Airport', 'Terminal', 'Tram', 'Bridge', 'Bus Station', 'Ferry', 'Hotel', 'Antique Hotel', 'Bed & Breakfast', 'Hostel', 'Luxury Hotel', 'Modern Hotel', 'Motel', 'Light Rail', 'Rental Car', 'Resort', 'Subway', 'Train Station', 'Travel-Other']

def get_cate():
    sources = ['Community', 'Entertainment', 'Food', 'Nightlife', 'Outdoors', 'Shopping', 'Travel']
    dests = ['Community', 'Campus Spot', 'Administration', 'Campus Commons', 'Campus - Other', 'Dormitory', 'Frat House', 'Hall', 'Lab', 'Rec Center', 'Sorority House', 'Student Center', 'Government', 'City Hall', 'Courthouse', 'Fire Station', 'Police Station', 'Post Office', 'Home', 'Apartment', 'Condo', 'Craftsman', 'Duplex', 'Modern', 'Victorian', 'Library', 'Office', 'Corporate Office', 'Coworking', 'Non-Profit', 'Skyscraper', 'Startup', 'Warehouse & Industrial', 'Place of Worship', 'Church', 'Contemporary Church', 'Historic Church', 'Mission', 'Mosque', 'Other - Place of Worship', 'Synagogue', 'Temple', 'School', 'College', 'High School', 'Law School', 'Medical School', 'School - Other', 'Trade/Tech School', 'University', 'Entertainment', 'Aquarium', 'Arcade', 'Arena or Stadium', 'Arena', 'Baseball Stadium', 'Soccer Stadium', 'Stadium', 'Art', 'Asian Art Museum', 'Gallery', 'Modern Art Museum', 'Sculpture', 'Traditional Art Museum', 'Art & Culture', 'Bowling', 'Casino', 'Convention Center', 'Disc Golf Course', 'Golf Course', 'Ice Skating', 'Live Music', 'Local Sports', 'Baseball Field', 'Basketball Court', 'Football Field', 'Soccer Field', 'Tennis Court', 'Movie Theatre', 'Museum', 'History Museum', 'Planetarium', 'Science Museum', 'Other - Entertainment', 'Performing Arts', 'Racetrack', 'Skatepark', 'Theatre', 'Theme Park', 'Zoo', 'Aviary', 'Bears', 'Elephants', 'Giraffes', 'Hippos', 'Lions', 'Other - Zoo', 'Primates', 'Reptiles', 'Tigers', 'Zebras', 'Food', 'African', 'American', 'Asian', 'Asian Food', 'Pan Pacific', 'Sushi', 'Thai', 'Vietnamese', 'Bakery', 'BBQ', 'Breakfast', 'Burgers', 'Coffee Shop', 'Dessert', 'Candy Store', 'Chocolate', 'Frozen Yogurt', 'Ice Cream', 'Snow Cones', 'Diner', 'Doughnuts', 'Fine Dining', 'Fish & Chips', 'French', 'Indian', 'Italian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Other - Food ', 'Pizza', 'Sandwich Shop', 'Seafood', 'Soul Food', 'South American/Latin', 'Steakhouse', 'Street Fare', 'Hot Dogs & Sausages', 'Other - Street Fare', 'Snow Cones', 'Tacos', 'Tea Room', 'Vegetarian', 'Vineyard', 'Wings', 'Nightlife', 'Bar', 'Dancefloor', 'Dive Bar', 'Karaoke', 'Microbrewery', 'Other - Nightlife ', 'Pub', 'Saloon', 'Sports Bar', 'Ultra-lounge', 'Wine Bar', 'Outdoors', 'Architecture', 'Castle', 'Historic Landmark', 'Lighthouse', 'Monument', 'Other - Architecture', 'Tower', 'Beach', 'Canal', 'Cave', 'Cemetery', 'Dog Park', 'Farm', 'Fountain', 'Garden', 'Lake/River', 'Park', 'Campground', 'City Park', 'National Park', 'Park - Other', 'Regional/State Park', 'Trailhead', 'Pavilion', 'Playground', 'Plaza/Square', 'Pool', 'Scenic Lookout', 'Ski & Snowboard Area', 'Chalet', 'Glade', 'Gondola', 'Halfpipe', 'Lifthouse', 'Summit', 'Terrain Park', 'Vineyard', 'Shopping', 'Antiques', 'Apparel', 'Accessories', "Children's Apparel", "Men's Apparel", 'Shoes', "Women's Apparel", 'Bank & Financial', 'Bookstore', 'Crafts & Creative', 'Department Store', 'Discount Store', 'Drugstore & Pharmacy', 'Farmers Market', 'Flower Shop', 'Furniture', 'Gas & Automotive', 'Grocery', 'Gym', 'Hardware Store', 'Health & Fitness', 'Mall', 'Medical', 'Dentist', "Doctor's Office", 'Drugstore & Pharmacy', 'Hospital', 'Other - Medical', 'Movie Rental', 'Music', 'Other - Services', 'Other - Shopping', 'Paper Goods', 'Pet Store', 'Record Store', 'Salon & Barbershop', 'Sports & Outdoors', 'Bike Shop', 'Camping & Outdoors', 'Golf Shop', 'Other Sports', 'Running', 'Skate Shop', 'Ski Shop', 'Tattoo Parlor', 'Technology', 'Thrift Store', 'Tobacco & Cigars', 'Toys & Games', 'Veterinarian', 'Wine & Spirits', 'Travel', 'Airport', 'Terminal', 'Tram', 'Bridge', 'Bus Station', 'Ferry', 'Hotel', 'Antique Hotel', 'Bed & Breakfast', 'Hostel', 'Luxury Hotel', 'Modern Hotel', 'Motel', 'Light Rail', 'Rental Car', 'Resort', 'Subway', 'Train Station', 'Travel-Other']

    cate_dict = {}

    for source in sources:
        cate_dict.update({source: []})

    counter = 0

    for dest in dests:
        if dest in sources:
            counter += 1
        cate_dict[sources[counter-1]].append(dest)

    total = 0
    for k in cate_dict.keys():
        print("key: %s, list length: %d"%(k, len(cate_dict[k])))
        total += len(cate_dict[k])
        print(cate_dict[k])

    print("total length: %d, dest length: %d"%(total, len(dests)))

    with open('cate_dict.pkl', 'wb') as f:
        pickle.dump(cate_dict, f)

# key: Community, list length: 50
# key: Entertainment, list length: 51
# key: Food, list length: 46
# key: Nightlife, list length: 12
# key: Outdoors, list length: 38
# key: Shopping, list length: 52
# key: Travel, list length: 20

'''
key: Community, len: 468653
key: Entertainment, len: 124165
key: Food, len: 610093
key: Nightlife, len: 114250
key: Outdoors, len: 171285
key: Shopping, len: 659735
key: Travel, len: 178464
'''

'''
time transfer:
2009-2010作为预训练数据
2011作为下游任务数据

field transfer:
Community类别做预训练数据
Entertainment, Food和Shopping类别做下游任务数据

time+field transfer:
2009-2010年的Community类别做预训练数据
2011年的Entertainment, Food和Shopping类别做下游任务数据
'''


# gowalla_checkins.csv打卡数据
# gowalla_spots_subset1.csv id对应类别的数据
# gowalla_spots_subset2.csv 经纬度数据
# gowalla_category_structure.json 类别嵌套的json数据

# gowalla_processed.csv 打卡数据处理为时间戳作为记录时间
# gowalla_spot_cate.csv 转换到对应类别
