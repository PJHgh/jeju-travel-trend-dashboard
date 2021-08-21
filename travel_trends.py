import altair as alt
import pydeck as pdk
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from datetime import datetime
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_vega_lite import vega_lite_component, altair_component

from utils import *

st.set_page_config(page_title='Travel Trends on Instagram', layout="wide")

st.title('Travel Trends on Instagram')


############################################################### define function


@st.cache(persist=True)
def load_data():
    dir_path = './data/'
    csv_paths = list(map(lambda x:dir_path+x, os.listdir(dir_path)))

    df = None
    for path in csv_paths:
        if df is None:
            df = pd.read_csv(path)
        else:
            df = pd.concat([df, pd.read_csv(path)])

    for path in csv_paths:
        os.remove(path)
    df.to_csv(dir_path+'all.csv', index=False)
    df = preprocessing(df)
    
    if len(csv_paths) > 1:
        insta_places_info = get_and_update_insta_places_info(df)
    else:
        insta_places_info = get_insta_places_info()
    
    insta_places_info['lat'] = insta_places_info['lat'].apply(np.float64)
    insta_places_info['lon'] = insta_places_info['lon'].apply(np.float64)
    insta_places_info = insta_places_info[insta_places_info['lat'] <= 33.931441]
    insta_places_info = insta_places_info[insta_places_info['lat'] >= 32.731441]
    insta_places_info = insta_places_info[insta_places_info['lon'] <= 127.874237]
    insta_places_info = insta_places_info[insta_places_info['lon'] >= 125.259003]
    return df, insta_places_info

@st.cache(persist=True)
def load_static_df(df, insta_places_info):
    # tag2loc = get_tag2loc(df)
    top_k_tags, top_k_counts = get_top_k_tags(df, k=20)
    top_k_tag_place = []
    name_count = 0
    for tag, count in zip(top_k_tags, top_k_counts):
        # if tag in tag2loc.keys():
        #     name, lat, lon, place = tag2loc[tag]
        #     top_k_tag_place.extend([[name, lat, lon, tag]]*count)
        #     name_count += 1
        #     continue

        try:
            name, lat, lon, place = get_lat_and_lon(tag[1:])
            top_k_tag_place.extend([[name, lat, lon, tag]]*count)
            name_count += 1
        except:
            pass

        if name_count >= 10:
            break
        
    top_k_tags_df = pd.DataFrame(top_k_tag_place)
    top_k_tags_df.columns = ['name', 'lat', 'lon', 'tag']
    top_k_tags_df['lat'] = top_k_tags_df['lat'].apply(np.float64)
    top_k_tags_df['lon'] = top_k_tags_df['lon'].apply(np.float64)

    apriori_df = assciation_analysis(df, min_support=0.003)

    # tag_df = get_tag_df(df, tag2loc)
    # tag_info_df = tag_df[['date', 'tag_info']]
    # tag_info_df['name'] = tag_info_df['tag_info'].apply(lambda x:list(x)[0])
    # tag_info_df['lat'] = tag_info_df['tag_info'].apply(lambda x:list(x)[1].astype(np.float64))
    # tag_info_df['lon'] = tag_info_df['tag_info'].apply(lambda x:list(x)[2].astype(np.float64))

    place_df = get_place_df(df, insta_places_info)
    place_info_df = place_df[['date', 'place_info']]
    place_info_df['name'] = place_info_df['place_info'].apply(lambda x:list(x)[0])
    place_info_df['lat'] = place_info_df['place_info'].apply(lambda x:list(x)[1].astype(np.float64))
    place_info_df['lon'] = place_info_df['place_info'].apply(lambda x:list(x)[2].astype(np.float64))
    # return top_k_tags_df, apriori_df, tag_info_df, place_info_df

    return top_k_tags_df, apriori_df, place_info_df, place_df

@st.cache(allow_output_mutation=True)
def load_period_df(df, place_info_df):
    period_list = get_period_list()
    period_top_k_tags = get_period_top_k_tags(df, k=10)
    
    period_top_k_tags_list = []
    for date in period_list[1:]:
        top_k_tags_count = period_top_k_tags.T[f'{date}_count']
        
        for tag, count in zip(top_k_tags_count.index, top_k_tags_count.values):
            try:
                name, lat, lon, place = get_lat_and_lon(tag[1:])
                period_top_k_tags_list.extend([[date, name, lat, lon, tag]]*int(count))
            except:
                # stoptags = get_pickle('./etc/stoptags')
                # stoptags += [tag]
                # save_pickle('./etc/stoptags', stoptags)
                pass
        
    period_top_k_tags_list = pd.DataFrame(period_top_k_tags_list)
    period_top_k_tags_list.columns = ['date', 'name', 'lat', 'lon', 'tag']
    period_top_k_tags_list['lat'] = period_top_k_tags_list['lat'].apply(np.float64)
    period_top_k_tags_list['lon'] = period_top_k_tags_list['lon'].apply(np.float64)

    periopd_top_k_place_df = get_periopd_top_k_place_df(place_info_df)
    return period_list, period_top_k_tags, period_top_k_tags_list, periopd_top_k_place_df

@st.cache()
def run_elastic_search(place_df):
    index_name='kakao'
    corpus = get_corpus(place_df)

    es = set_index_and_server(index_name)
    populate_index(es_obj=es, index_name=index_name, evidence_corpus=corpus)


def load_map(data, lat, lon, zoom):
    layer1 = pdk.Layer(
            "CPUGridLayer",
            data=data,
            get_position=["lon", "lat"],
            get_radius=50,
            get_fill_color='[255, 255, 255]',
            # radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            auto_highlight=True,
            extruded=True,
            cellSizePixels=10,
            bearing = 0,
            pitch = 0,
            )

    view_state = {"bearing": 0, 
                  "latitude": lat, 
                  "longitude": lon,
                  "maxZoom": 20,
                  "minZoom": 0, 
                  "pitch": 50, 
                  "zoom": zoom,
                  }

    r = pdk.Deck(
        # map_style="mapbox://styles/mapbox/light-v9",
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state=view_state,
        layers=[layer1],
        # width='100%',
        # height=500,
        # tooltip=True,
        )
    st.write(r)


def load_recommender_map(recommender_df):
    min_support = 1/(recommender_df['distance'].min()+1e-8)
    max_support = 1/(recommender_df['distance'].max()+1e-8)
    recommender_df['scaled_distance'] = (1/(recommender_df['distance']+1e-8) - min_support)/(max_support - min_support)
    recommender_df['scaled_distance'] = recommender_df['scaled_distance']*(1-0.5)+0.5
    recommender_df['start_lat'] = recommender_df['start_lat'].apply(np.float64)
    recommender_df['start_lon'] = recommender_df['start_lon'].apply(np.float64)
    recommender_df['end_lat'] = recommender_df['end_lat'].apply(np.float64)
    recommender_df['end_lon'] = recommender_df['end_lon'].apply(np.float64)

    layer1 = pdk.Layer(
        'ArcLayer',
        recommender_df,
        get_source_position='[start_lon, start_lat]',
        get_target_position='[end_lon, end_lat]',
        get_width='1 + 2 * scaled_distance',
        get_source_color='[255, 255, 120]',
        get_target_color='[255, 0, 0]',
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.data_utils.compute_view(recommender_df[['start_lon', 'start_lat']].values)
    view_state.zoom = 10
    view_state.bearing = -15
    view_state.pitch = 45

    r = pdk.Deck(map_style="mapbox://styles/mapbox/dark-v9",
                    layers=[layer1],
                    initial_view_state=view_state)
    st.write(r)


############################################################### data load

df, insta_places_info = load_data()
# top_k_tags_df, apriori_df, tag_info_df, place_info_df = load_static_df(df, insta_places_info)
top_k_tags_df, apriori_df, place_info_df, place_df = load_static_df(df, insta_places_info)
period_list, period_top_k_tags, period_top_k_tags_list, periopd_top_k_place_df = load_period_df(df, place_info_df)
run_elastic_search(place_df)

st.markdown("---")

############################################################### [Tag] Frequency Analysis

col1, col2 = st.beta_columns((3, 2))

with col1:
    st.header("[Tag] Frequency Analysis")
    
    zoom_level = 9.5
    midpoint = (np.average(top_k_tags_df["lat"]), np.average(top_k_tags_df["lon"]))
    load_map(top_k_tags_df, midpoint[0], midpoint[1], zoom_level)
    
with col2:
    st.subheader(f"Histogram")
    r = px.histogram(top_k_tags_df, x="name").update_xaxes(type="category", categoryorder="total descending")
    st.write(r)

# col1, col2 = st.beta_columns((3, 2))

# with col1:
#     st.subheader(f"Select Day")
#     selected_day_tag = st.slider(" ", period_list[-1], period_list[1])
#     top_k_tags_df = tag_info_df[tag_info_df['date'] == selected_day_tag]
    
#     zoom_level = 9.5
#     midpoint = (np.average(top_k_tags_df["lat"]), np.average(top_k_tags_df["lon"]))
#     load_map(top_k_tags_df, midpoint[0], midpoint[1], zoom_level)
    
# with col2:
#     st.subheader(f"Histogram - {selected_day_tag}")

#     r = px.histogram(top_k_tags_df, x="name").update_xaxes(type="category", categoryorder="total descending")
#     st.write(r)

st.markdown("---")


############################################################### [Place] Frequency Analysis

col1, col2 = st.beta_columns((3, 2))

top10_name = list(map(lambda x:x[0], sorted(place_info_df.groupby(['name'])['name'].count().items(), key=lambda x:x[1], reverse=True)[:10]))
top10_df = place_info_df[place_info_df['name'].apply(lambda x:x in top10_name)]

with col1:
    st.header("[Place] Frequency Analysis")

    zoom_level = 9.5
    midpoint = (np.average(top10_df["lat"]), np.average(top10_df["lon"]))
    load_map(top10_df, midpoint[0], midpoint[1], zoom_level)
    
with col2:
    st.subheader(f"Histogram")
    r = px.histogram(top10_df, x="name").update_xaxes(type="category", categoryorder="total descending")
    st.write(r)

# col1, col2 = st.beta_columns((3, 2))

# with col1:
#     st.subheader(f"Select Day")
#     selected_day_place = st.slider("  ", period_list[-1], period_list[1])
#     period_place_info_df = place_info_df[place_info_df['date'] == selected_day_place]

#     top10_name = list(map(lambda x:x[0], sorted(period_place_info_df.groupby(['name'])['name'].count().items(), key=lambda x:x[1], reverse=True)[:10]))
#     top10_df = period_place_info_df[period_place_info_df['name'].apply(lambda x:x in top10_name)]
    
#     zoom_level = 9.5
#     midpoint = (np.average(top10_df["lat"]), np.average(top10_df["lon"]))
#     load_map(top10_df, midpoint[0], midpoint[1], zoom_level)
    
# with col2:
#     st.subheader(f"Histogram - {selected_day_place}")

#     r = px.histogram(top10_df, x="name").update_xaxes(type="category", categoryorder="total descending")
#     st.write(r)

st.markdown("---")


############################################################### [Tag] Frequency Differential Analysis

period_top_k_tags = period_top_k_tags.T[list(map(str, period_list[::-1][:-1]))].T
period_top_k_tags.index = map(lambda x:datetime.strptime(str(x), '%Y%m%d'), period_list[::-1][:-1])
period_top_k_tags.columns = list(period_top_k_tags.columns)

col1, col2 = st.beta_columns((3, 2))

with col1:
    st.header("[Tag] Frequency Differential Analysis")

    selected_day_tag = st.slider("Select Day  ", period_list[-1], period_list[1])
    top_k_tags_df = period_top_k_tags_list[period_top_k_tags_list['date'] == selected_day_tag]
    
    zoom_level = 9.5
    midpoint = (np.average(top_k_tags_df["lat"]), np.average(top_k_tags_df["lon"]))
    load_map(top_k_tags_df, midpoint[0], midpoint[1], zoom_level)
    
with col2:
    st.subheader(f"Histogram & Differential Frequency - {selected_day_tag}")

    st.line_chart(period_top_k_tags)

    r = px.histogram(top_k_tags_df, x="name").update_xaxes(type="category", categoryorder="total descending")
    st.write(r)
    
st.markdown("---")


############################################################### [Place] Frequency Differential Analysis

periopd_top_k_place_df.index = map(lambda x:datetime.strptime(str(x), '%Y%m%d'), period_list[::-1][:-1])
periopd_top_k_place_df.columns = list(periopd_top_k_place_df.columns)

col1, col2 = st.beta_columns((3, 2))

with col1:
    st.header("[Place] Frequency Differential Analysis")

    selected_day_place = st.slider("Select Day   ", period_list[-1], period_list[1])
    period_place_info_df = place_info_df[place_info_df['date'] == selected_day_place]

    top10_df = period_place_info_df[period_place_info_df['name'].apply(lambda x:x in list(periopd_top_k_place_df.columns))]
    
    zoom_level = 9.5
    midpoint = (np.average(top10_df["lat"]), np.average(top10_df["lon"]))
    load_map(top10_df, midpoint[0], midpoint[1], zoom_level)
    
with col2:
    st.subheader(f"Histogram & Differential Frequency - {selected_day_place}")

    st.line_chart(periopd_top_k_place_df)

    r = px.histogram(top10_df, x="name").update_xaxes(type="category", categoryorder="total descending")
    st.write(r)

st.markdown("---")

############################################################### Association Rules Analysis

assciation_df = get_assciation_df(apriori_df)
min_support = assciation_df['support'].min()
max_support = assciation_df['support'].max()
assciation_df['scaled_support'] = (assciation_df['support'] - min_support)/(max_support - min_support)
assciation_df['scaled_support'] = assciation_df['scaled_support']*(1-min_support)+min_support
assciation_df['start_lat'] = assciation_df['start_lat'].apply(np.float64)
assciation_df['start_lon'] = assciation_df['start_lon'].apply(np.float64)
assciation_df['end_lat'] = assciation_df['end_lat'].apply(np.float64)
assciation_df['end_lon'] = assciation_df['end_lon'].apply(np.float64)

layer1 = pdk.Layer(
    'ArcLayer',
    assciation_df,
    get_source_position='[start_lon, start_lat]',
    get_target_position='[end_lon, end_lat]',
    get_width='1 + 30 * scaled_support',
    get_source_color='[255, 255, 120]',
    get_target_color='[255, 0, 0]',
    pickable=True,
    auto_highlight=True
)

view_state = pdk.data_utils.compute_view(assciation_df[['start_lon', 'start_lat']].values)
view_state.zoom = 9
view_state.bearing = -15
view_state.pitch = 45

r = pdk.Deck(map_style="mapbox://styles/mapbox/dark-v9",
                layers=[layer1],
                initial_view_state=view_state)

st.header("Association Rules Analysis")
col1, col2 = st.beta_columns((3, 2))

with col1:
    st.write(r)

with col2:
    st.table(assciation_df[['start_name', 'end_name', 'support']][:11])
st.markdown("---")


############################################################### Recommender system

es, index_name = elastic_setting()

st.header("Recommender System")

keyword = st.text_input("Enter keyword", "Type Here ...")
if st.button('Submit'):
    try:
        name, lat, lon = recommender(es, index_name, keyword, place_df)
        recommender_df = get_recommender_df(assciation_df, name, lat, lon)

        st.success(name)
        
        col1, col2 = st.beta_columns((3, 2))

        with col1:
            load_recommender_map(recommender_df)

        with col2:
            st.subheader(f"Good places to go with {name}")
            st.table(recommender_df[['start_name', 'end_name', 'distance']])
    except:
        st.error("Please enter a different keyword")

st.markdown("---")