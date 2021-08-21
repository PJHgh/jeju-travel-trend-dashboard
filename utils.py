import os
import re
import time
import heapq
import pickle
import requests
import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib import rc
from apyori import apriori
from collections import Counter
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def get_period_list():
    period_list = []
    # d = dt.date(2021, 7, 30)
    # t = dt.time(0, 0, 0)
    # now = dt.datetime.combine(d, t)
    for i in range(10):
        period = datetime.now() - relativedelta(days=i)
        # period = now - relativedelta(days=i)
        period = int(period.strftime('%Y%m%d'))
        period_list.append(period)
    return period_list


def preprocessing(df):
    '''
    1. 불필요 column 제거
    2. 데이터 내의 불용어 제거
    3. 중복 제거
    4. reset index

    date column 전처리
    1. int형 변환
    2. 최근 10일간 데이터만 남김

    '''

    # df = df.drop(['Unnamed: 0'], axis=1)
    for column in list(df.columns):
        if df[column].dtype == 'int64':
            continue
        df[column] = df[column].apply(lambda x:re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", str(x)))

    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop=True)

    df = df[df['date'] != 'nan']
    df['date'] = df['date'].apply(lambda x:int(''.join(str(x).split('-'))))

    recent = datetime.now() - relativedelta(days=10)
    recent = int(recent.strftime('%Y%m%d'))
    df = df[df['date'] >= recent]
    df = df.reset_index(drop=True)
    return df


def save_pickle(save_path, dataset):
    file = open(save_path, "wb")
    pickle.dump(dataset, file)
    file.close()


def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

### tag

def assciation_analysis(df, min_support=0.004):
    '''
    Tags 연관성 분석

    '''
    tags = df[df['tags'].apply(lambda x: True if len(x[1:-1]) > 0 else False)]['tags']
    tags = tags.apply(lambda x:list(map(lambda x:x.split("'")[1].strip(), x.split(','))))

    stoptags = get_pickle('./etc/stoptags')
    tags = tags.apply(lambda x:list(set(x) - set(stoptags)))
    tags = tags[tags.apply(len) > 0]

    result = list(apriori(list(tags), min_support=min_support))
    apriori_df = pd.DataFrame(result)
    apriori_df['length'] = apriori_df['items'].apply(len)
    apriori_df = apriori_df[apriori_df['length'] == 2].sort_values(by='support', ascending=False)
    return apriori_df

def get_info(x, place2info, kind):
    try:
        name, lat, lon = place2info[x]
    except:
        return ''
    
    if kind == 'name':
        return name
    elif kind == 'lat':
        return lat
    elif kind == 'lon':
        return lon
    elif kind == 'info':
        return [name, lat, lon]

def get_assciation_df(apriori_df):
    assciation_df = apriori_df[['items', 'support']].reset_index(drop=True)
    assciation_df['start_name'] = assciation_df['items'].apply(lambda x:list(x)[0][1:])
    assciation_df['end_name'] = assciation_df['items'].apply(lambda x:list(x)[1][1:])
    places = list(set(list(assciation_df['start_name'])+list(assciation_df['end_name'])))

    place2info = {}
    for place in places:
        try:
            name, lat, lon, place = get_lat_and_lon(place)
            place2info[place] = [name, lat, lon]
        except:
            pass

    assciation_df['start_lat'] = assciation_df['start_name'].apply(lambda x:get_info(x, place2info, 'lat'))
    assciation_df['start_lon'] = assciation_df['start_name'].apply(lambda x:get_info(x, place2info, 'lon'))
    assciation_df['start_name'] = assciation_df['start_name'].apply(lambda x:get_info(x, place2info, 'name'))

    assciation_df['end_lat'] = assciation_df['end_name'].apply(lambda x:get_info(x, place2info, 'lat'))
    assciation_df['end_lon'] = assciation_df['end_name'].apply(lambda x:get_info(x, place2info, 'lon'))
    assciation_df['end_name'] = assciation_df['end_name'].apply(lambda x:get_info(x, place2info, 'name'))

    assciation_df = assciation_df[assciation_df['start_name'] != assciation_df['end_name']]
    assciation_df = assciation_df[(assciation_df['start_name'] != '') & (assciation_df['end_name'] != '')]
    assciation_df = assciation_df.pivot_table(index=['start_name', 'end_name', 'start_lat', 'start_lon', 'end_lat', 'end_lon'], values='support', aggfunc='sum')
    assciation_df = assciation_df.reset_index([0,1,2,3,4,5])
    assciation_df = assciation_df.sort_values(by='support', ascending=False)
    assciation_df = assciation_df.reset_index(drop=True)
    return assciation_df

def get_top_k_tags(df, k=10):
    '''
    빈도수 분석
    전체 데이터 중 Top k개 tag 추출

    '''
    tags = df[df['tags'].apply(lambda x: True if len(x[1:-1]) > 0 else False)]['tags']
    tags = tags.apply(lambda x:list(map(lambda x:x.split("'")[1].strip(), x.split(','))))

    all_tags = tags.sum()
    all_tags_counter = Counter(all_tags)

    stoptags = get_pickle('./etc/stoptags')
    all_tags_list = list(set(all_tags_counter.keys()) - set(stoptags))

    all_tags_counter = [(-1*all_tags_counter[key], key) for key in all_tags_list]
    heapq.heapify(all_tags_counter)

    top_k_tags = []
    top_k_counts = []
    while len(top_k_tags) < k:
        try:
            count, tag = heapq.heappop(all_tags_counter)
            top_k_tags.append(tag)
            top_k_counts.append(-1*count)
        except:
            print(f'list length == {len(top_k_tags)} != {k}')
            break
    return top_k_tags, top_k_counts

def get_period_top_k_tags(df, k=10):
    '''
    빈도수 변화 분석
    날짜 별 데이터 중 Top k개 tag 추출

    '''
    period_list = get_period_list()

    top_k_df = None
    for max_date, min_date in zip(period_list[:-1], period_list[1:]):
        period_df = df[df['date'] == min_date]
        period_df = period_df.reset_index(drop=True)

        if len(period_df) > 0:
            top_k_tags, top_k_counts = get_top_k_tags(period_df, k=300)
        else:
            top_k_tags, top_k_counts = [], []
        
        if top_k_df is None:
            top_k_df = pd.DataFrame({'tags':top_k_tags, f'{str(min_date)}_count':top_k_counts})
            top_k_df[str(min_date)] = top_k_df[f'{str(min_date)}_count'].apply(lambda x: x/len(period_df)*100)
        else:
            top_k_df = pd.merge(pd.DataFrame({'tags':top_k_tags, f'{str(min_date)}_count':top_k_counts}), top_k_df, how='outer', on='tags')
            top_k_df[str(min_date)] = top_k_df[f'{str(min_date)}_count'].apply(lambda x: x/len(period_df)*100)
            top_k_df = top_k_df[top_k_df[f'{str(max_date)}_count'].isnull() == False]
    top_k_df = top_k_df[top_k_df[f'{str(period_list[-1])}_count'].isnull() == False]
    top_k_df = top_k_df.set_index('tags')
    
    diff_mean = top_k_df.T.diff().mean()
    top_k_df['mean'] = diff_mean.T
    top_k_df = top_k_df.sort_values(by='mean', ascending=False)
    top_k_df = top_k_df.drop(['mean'], axis=1)
    return top_k_df.iloc[:k].T

def get_tag2loc(df):

    tags = df[df['tags'].apply(lambda x: True if len(x[1:-1]) > 0 else False)]['tags']
    tags = tags.apply(lambda x:list(map(lambda x:x.split("'")[1].strip(), x.split(','))))

    all_tags = tags.sum()
    all_tags_counter = Counter(all_tags)

    stoptags = get_pickle('./etc/stoptags')
    all_tags_list = list(set(all_tags_counter.keys()) - set(stoptags))

    insta_places_info = pd.read_csv('./etc/insta_places_info.csv')
    insta_places_info['insta_names'] = insta_places_info['insta_names'].apply(lambda x:list(map(lambda x:re.sub(r"[']", "", x).strip(), x[1:-1].split(','))))
    searched_places = set(insta_places_info['insta_names'].sum())

    tag2loc = {}
    tag2loc_df = []
    for key in all_tags_list:
        tag, count = key, all_tags_counter[key]
        if tag[1:] in searched_places:
            name, lat, lon = insta_places_info[insta_places_info['insta_names'].apply(lambda x:tag[1:] in x)][['name', 'lat', 'lon']].iloc[0]
            tag2loc[tag] = [name, lat, lon, tag[1:]]
            tag2loc_df.append([name, lat, lon, tag[1:]])
            continue

        try:
            name, lat, lon, place = get_lat_and_lon(tag[1:])
            tag2loc[tag] = [name, lat, lon, tag[1:]]
            tag2loc_df.append([name, lat, lon, tag[1:]])
        except:
            pass

    tag2loc_df_ = pd.DataFrame(tag2loc_df)
    tag2loc_df_.columns = ['name', 'lat', 'lon', 'insta_name']
    tag2loc_df_['count'] = tag2loc_df_['insta_name'].apply(lambda x:all_tags_counter['#'+x])
    tag2loc_df_['insta_name'] = tag2loc_df_['insta_name'].apply(lambda x:[x])
    
    insta_places_info = insta_places_info.set_index(['name', 'lat', 'lon'])
    tag2loc_df = tag2loc_df_.pivot_table(index=['name', 'lat', 'lon'], values='count', aggfunc='sum')
    tag2loc_df['insta_names'] = tag2loc_df_.pivot_table(index=['name', 'lat', 'lon'], values='insta_name', aggfunc='sum')
    
    new_insta_places_info_ = pd.concat([insta_places_info, tag2loc_df])
    new_insta_places_info = new_insta_places_info_.pivot_table(index=['name', 'lat', 'lon'], values='count', aggfunc='sum')
    new_insta_places_info['insta_names'] = new_insta_places_info_.pivot_table(index=['name', 'lat', 'lon'], values='insta_names', aggfunc='sum')
    new_insta_places_info.to_csv('./etc/insta_places_info.csv')
    return tag2loc


def get_tag_df(df, tag2loc):
    stoptags = get_pickle('./etc/stoptags')

    tag_df = df[df['tag'].apply(lambda x:False if x.strip() in stoptags else True)]
    tag_df['tag_info'] = tag_df['tag'].apply(lambda x:get_info(x, tag2loc, 'info'))
    tag_df = tag_df[tag_df['tag_info'] != '']
    tag_df = tag_df.reset_index(drop=True)
    return tag_df


### place


def get_lat_and_lon(place):
    insta_places_info = pd.read_csv('./etc/insta_places_info.csv')
    insta_places_info['insta_names'] = insta_places_info['insta_names'].apply(lambda x:list(map(lambda x:re.sub(r"[']", "", x).strip(), x[1:-1].split(','))))
    searched_places = set(insta_places_info['insta_names'].sum())

    if place.strip() in searched_places:
        name, lat, lon = insta_places_info[insta_places_info['insta_names'].apply(lambda x:place.strip() in x)][['name', 'lat', 'lon']].iloc[0]
        return [name, lat, lon, place.strip()]

    url = f'https://dapi.kakao.com/v2/local/search/keyword.json?query=제주도 {place}'
    headers = {"Authorization":"******"}

    places = requests.get(url, headers=headers).json()['documents']
    place_loc = places[0]
    name = place_loc['place_name']
    lon, lat = place_loc['x'], place_loc['y']
    
    return [name, lat, lon, place]

def get_and_update_insta_places_info(df):
    places = df[df['place'] != 'nan']['place']
    places = places.apply(lambda x:[str(x).strip()])
    places = places.sum()
    places = [[k, v] for k, v in Counter(places).items()]
    places_df = pd.DataFrame(places)
    places_df.columns = ['insta_name', 'count']

    places = list(places_df['insta_name'])
    heapq.heapify(places)

    insta_places_info = pd.read_csv('./etc/insta_places_info.csv')
    insta_places_info['insta_names'] = insta_places_info['insta_names'].apply(lambda x:list(map(lambda x:re.sub(r"[']", "", x).strip(), x[1:-1].split(','))))
    searched_places = set(insta_places_info['insta_names'].sum())

    locations = []
    while places:
        place = heapq.heappop(places)
        if place.strip() in searched_places:
            name, lat, lon = insta_places_info[insta_places_info['insta_names'].apply(lambda x:place.strip() in x)][['name', 'lat', 'lon']].iloc[0]
            locations.append([name, lat, lon, place.strip()])
            continue

        try:
            locations.append(get_lat_and_lon(place))
            time.sleep(0.1)
        except:
            pass
    
    locations_df = pd.DataFrame(locations)
    locations_df.columns = ['name', 'lat', 'lon', 'insta_name']
    locations_df['count'] = locations_df['insta_name'].apply(lambda x:list(places_df[places_df['insta_name'] == x]['count'])[0])
    locations_df['insta_name'] = locations_df['insta_name'].apply(lambda x:[x])
    
    insta_places_info = insta_places_info.set_index(['name', 'lat', 'lon'])
    locations_data = locations_df.pivot_table(index=['name', 'lat', 'lon'], values='count', aggfunc='sum')
    locations_data['insta_names'] = locations_df.pivot_table(index=['name', 'lat', 'lon'], values='insta_name', aggfunc='sum')

    new_insta_places_info_ = pd.concat([insta_places_info, locations_data])
    new_insta_places_info = new_insta_places_info_.pivot_table(index=['name', 'lat', 'lon'], values='count', aggfunc='sum')
    new_insta_places_info['insta_names'] = new_insta_places_info_.pivot_table(index=['name', 'lat', 'lon'], values='insta_names', aggfunc='sum')
    new_insta_places_info.to_csv('./etc/insta_places_info.csv')
    return locations_data.reset_index([0,1,2])

def get_place_df(df, insta_places_info):
    stoptags = get_pickle('./etc/stoptags')
    stopplace = list(map(lambda x:x[1:], stoptags))

    place_df = df[df['place'].apply(lambda x:False if x.strip() in stopplace else True)]
    place_df['place_info'] = place_df['place'].apply(lambda x:get_place_info(x, insta_places_info))
    place_df = place_df[place_df['place_info'] != ()]
    place_df = place_df.reset_index(drop=True)
    return place_df

def get_insta_places_info():
    insta_places_info = pd.read_csv('./etc/insta_places_info.csv')
    insta_places_info['insta_names'] = insta_places_info['insta_names'].apply(lambda x:list(map(lambda x:re.sub(r"[']", "", x).strip(), x[1:-1].split(','))))
    return insta_places_info

def get_place_info(place, insta_places_info):
    new_place = insta_places_info[insta_places_info['insta_names'].apply(lambda x:True if place in x else False)]
    try:
        name = new_place['name'].iloc[0]
        lat = new_place['lat'].iloc[0]
        lon = new_place['lon'].iloc[0]
        return (name, lat, lon)
    except:
        return ()

def get_periopd_top_k_place_df(place_info_df, k=10):
    period_list = get_period_list()

    periopd_top_k_place_df = None
    for date in period_list[1:]:
        period_place_info_df = place_info_df[place_info_df['date'] == date]
        period_place_info_df = period_place_info_df.reset_index(drop=True)

        place_info_count = period_place_info_df.groupby(['place_info'])['place_info'].count()

        period_place_info_df = []
        for (name, _, _), count in zip(list(place_info_count.index), list(place_info_count)):
            period_place_info_df.append([name, count])

        if periopd_top_k_place_df is None:
            periopd_top_k_place_df = pd.DataFrame(period_place_info_df)
            periopd_top_k_place_df.columns = ['name', str(date)]
            periopd_top_k_place_df[str(date)] = periopd_top_k_place_df[str(date)].apply(lambda x: x/len(periopd_top_k_place_df)*100)
        else:
            period_place_info_df = pd.DataFrame(period_place_info_df)
            period_place_info_df.columns = ['name', str(date)]
            period_place_info_df[str(date)] = period_place_info_df[str(date)].apply(lambda x: x/len(period_place_info_df)*100)

            periopd_top_k_place_df = pd.merge(period_place_info_df, periopd_top_k_place_df, how='outer', on='name')
            periopd_top_k_place_df = periopd_top_k_place_df[periopd_top_k_place_df[str(date)].isnull() == False]
    periopd_top_k_place_df = periopd_top_k_place_df[periopd_top_k_place_df[str(period_list[1])].isnull() == False]
    periopd_top_k_place_df = periopd_top_k_place_df.set_index('name')

    diff_mean = periopd_top_k_place_df.T.diff().mean()
    periopd_top_k_place_df['mean'] = diff_mean.T
    periopd_top_k_place_df = periopd_top_k_place_df.sort_values(by='mean', ascending=False)
    periopd_top_k_place_df = periopd_top_k_place_df.drop(['mean'], axis=1)

    periopd_top_k_place_df = periopd_top_k_place_df.iloc[:k].T
    return periopd_top_k_place_df


### Recommender system


def get_corpus(place_df):
    corpus = list(map(lambda x:' 장소: '.join(x), place_df[['content', 'place']].values))
    return corpus

def set_index_and_server(index_name) :
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    index_config = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict", 
            "properties": {
                "tag": {"type": "text", "analyzer": "nori_analyzer"}
                }
            }
        }

    print('elastic serach ping :', es.ping())
    print(es.indices.create(index=index_name, body=index_config, ignore=400))

    return es

def populate_index(es_obj, index_name, evidence_corpus):
    evidence_corpus = [{'tag':tag} for tag in evidence_corpus]

    for i, tag in enumerate(evidence_corpus):
        try:
            index_status = es_obj.index(index=index_name, id=i, body=tag)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')

def elastic_setting(index_name='kakao'):
    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])
    
    return es, index_name

def search_es(es_obj, index_name, keyword, n_results):
    # search query
    query = {
            'query': {
                'match': {
                    'tag': keyword
                    }
                }
            }
    # n_result => 상위 몇개를 선택?
    res = es_obj.search(index=index_name, body=query, size=n_results)
    
    return res

def elastic_retrieval(es, index_name, keyword, n_results=10):
    res = search_es(es, index_name, keyword, n_results)
    # 매칭된 context만 list형태로 만든다.
    context_list = list((hit['_source']['tag'], hit['_score']) for hit in res['hits']['hits'])
    return context_list

def recommender(es, index_name, keyword, place_df):
    search_result = elastic_retrieval(es, index_name, keyword)

    content_list = list(map(lambda x:x[0].split(' 장소: ')[0], search_result))
    search_df = place_df[place_df['content'].apply(lambda x:x in set(content_list))]
    search_df['elastic_score'] = search_df['content'].apply(lambda x:search_result[content_list.index(x)][1])
    search_df['num_of_rewiew'] = search_df['rewiew'].apply(lambda x:len(x))
    search_df['like'] = search_df['like'].apply(int)

    search_df = search_df.sort_values(by='elastic_score')
    search_df['elastic_rank'] = np.arange(len(search_df))

    search_df = search_df.sort_values(by='num_of_rewiew')
    search_df['review_rank'] = np.arange(len(search_df))

    search_df = search_df.sort_values(by='like')
    search_df['like_rank'] = np.arange(len(search_df))

    search_df['total_score'] = search_df.loc[:,['elastic_rank','review_rank','like_rank']].sum(axis=1)
    search_df = search_df.sort_values(by=['total_score', 'date'], ascending=False)

    name, lat, lon = search_df.iloc[0]['place_info']
    return name, lat, lon

def get_recommender_df(assciation_df, name, lat, lon):
    if 'start_distance' in assciation_df.columns:
        assciation_df = assciation_df.drop('start_distance', axis=1)
        
    assciation_df_ = assciation_df.drop_duplicates(keep='first')
    assciation_df_['start_distance'] = assciation_df_.loc[:,['start_lat', 'start_lon']].apply(lambda x:((lat-x[0])**2+(lon-x[1])**2)**0.5, axis=1)
    if assciation_df_['start_distance'].min() < 0.1:
        assciation_df_ = assciation_df_[assciation_df_['start_distance'] < 0.1]
    else:
        assciation_df_ = assciation_df_[assciation_df_['start_distance'] == assciation_df_['start_distance'].min()]

    recommender_df = assciation_df_.loc[:,['start_name', 'start_lat', 'start_lon', 'start_distance']].drop_duplicates(keep='first')
    recommender_df.columns = ['end_name', 'end_lat', 'end_lon', 'distance']

    recommender_df['start_name'] = name
    recommender_df['start_lat'] = lat
    recommender_df['start_lon'] = lon

    assciation_df_['distance'] = assciation_df_.loc[:,['start_lat', 'start_lon', 'end_lat', 'end_lon']].apply(lambda x:((x[0]-x[2])**2+(x[1]-x[3])**2)**0.5, axis=1)
    assciation_df_ = assciation_df_[['end_name', 'end_lat', 'end_lon', 'distance', 'start_name', 'start_lat', 'start_lon']]

    recommender_df = pd.concat([recommender_df, assciation_df_])
    recommender_df = recommender_df[recommender_df['end_name'] != recommender_df['start_name']]
    recommender_df = recommender_df.reset_index(drop=True)
    return recommender_df