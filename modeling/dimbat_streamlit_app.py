import pandas as pd
import numpy as np
import plotly.express as px  # interactive charts
import streamlit as st  

from traitlets import All

import ujson as json
from cmath import nan

st.set_page_config(
    page_title="DIMBAT - all we wont to write about it",
    page_icon=None,
    layout="wide",
)


def new_data(path="../data", year = 2011, disaster = 'Biological', location = 'Germany'):
    records1 = map(json.loads, open(path, encoding="utf8"))
    df = pd.DataFrame.from_records(records1)
    df['location'] = location
    df['disaster'] =  disaster
    df['year'] = year
    df = df.drop(['id'],axis=1)
    return df


df1 = new_data(path="/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/biological-ebola-2014.ndjson", year = 2014, disaster = 'Biological', location= 'nan')
df2 = new_data(path="/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/biological-mers-2014.ndjson", year = 2014, disaster = 'Biologocal', location=nan)
df3 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-bohol-2013.ndjson",year=2013, disaster = 'Earthquake',location='bohol')
df4 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-california-2013.ndjson",year=2013, disaster = 'Earthquake',location='california')
df5 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-chile-2013.ndjson",year=2013, disaster = 'Earthquake',location='chile')
df6 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-costarica-2012.ndjson",year=2012, disaster = 'Earthquake',location='costarica')
df7 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-guatemala-2012.ndjson",year=2012, disaster = 'Earthquake', location='guatemala')
df8 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-iraq-iran-2017.ndjson",year=2017, disaster = 'Earthquake',location='iran')
df9 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-italy-2012.ndjson",year=2012, disaster = 'Earthquake', location='italy')
df10 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-mexico-2017.ndjson",year=2017, disaster = 'Earthquake',location='mexico')
df11 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-nepal-2015.ndjson",year=2015, disaster = 'Earthquake',location='nepal')
df12 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-nepal-2018.ndjson",year=2018, disaster = 'Earthquake',location='nepal')
df13 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/earthquake-pakistan-2013.ndjson",year=2013, disaster = 'Earthquake',location='pakistan')
df14 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-alberta-2013.ndjson",year=2013, disaster = 'Flood',location='alberta')
df15 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-colorado-2013.ndjson",year=2013, disaster = 'Flood',location='colorado')
df16 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-india-2014.ndjson",year=2014, disaster = 'Flood', location='india')
df17 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-manila-2013.ndjson",year=2013, disaster = 'Flood',location='manila')
df18 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-pakistan-2014.ndjson",year=2014, disaster = 'Flood',location='pakistan')
df19 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-philipinnes-2012.ndjson",year=2012, disaster = 'Flood', location='philipinnes')
df20 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-queensland-2013.ndjson",year=2013, disaster = 'Flood',location='queensland')
df21 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-sardinia-2013.ndjson",year=2013, disaster = 'Flood', location='sardinia')
df22 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/flood-srilanka-2017.ndjson",year=2017, disaster = 'Flood',location='srilanka')
df23 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-hagupit-2014.ndjson",year=2014, disaster = 'Hurricane',location='hagupit')
df24 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-harvey-2017.ndjson",year=2017, disaster = 'Hurricane', location='harvey')
df25 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-irma-2017.ndjson",year=2017, disaster = 'Hurricane', location='iram')
df26 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-maria-2017.ndjson",year=2017, disaster = 'Hurricane', location= 'maria')
df27 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-pablo-2012.ndjson",year=2012, disaster = 'Hurricane', location='pablo')
df28 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-pam-2015.ndjson",year=2015, disaster = 'Hurricane', location='pam')
df29 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-sandy-2012.ndjson",year=2012, disaster = 'Hurricane',location='sandy')
df30 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-yolanda-2013.ndjson",year=2013, disaster = 'Hurricane', location='yolanda')
df31 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/hurricane-odile-2014.ndjson",year=2014, disaster = 'Hurricane',location='odile')
df32 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/industrial-savar-building-collapse-2013.ndjson",year=2013, disaster = 'Industrial',location='savar')
df33 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/industrial-texas-explosion-2013.ndjson",year=2013, disaster = 'Industrial',location='texas')
df34 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/industrial-venezuela-refinery-fire-2012.ndjson",year=2012, disaster = 'Industrial',location='venezula')
df35 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/other-russia-meteor-2013.ndjson",year=2013, disaster = 'Meteor',location='russia')
df36 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/other-singapore-haze-2013.ndjson",year=2013, disaster = 'Haze',location='singapore')
df37 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/societal-boston-bombing-2013.ndjson",year=2013, disaster = 'Societal',location='boston')
df38 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/societal-brazil-nightclub-fire-2013.ndjson",year=2013, disaster = 'Societal',location='brazil')
df39 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/societal-la-airport-shooting-2013.ndjson",year=2013, disaster = 'Societal',location='los angeles')
df40 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/tornado-joplin-2011.ndjson",year=2011, disaster = 'Tornado', location='joplin')
df41 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/tornado-oklahoma-2013.ndjson",year=2013, disaster = 'Tornado',location='oklahoma')
df42 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/transportation-glasgow-helicopter-crash-2013.ndjson",year=2013, disaster = 'Transportation', location='glasgow')
df43 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/transportation-la-train-crash-2013.ndjson",year=2013, disaster = 'Transportation',location='los angeles')
df44 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/transportation-ny-train-crash-2013.ndjson",year=2013, disaster = 'Transportation',location='newyork')
df45 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/transportation-spain-train-crash-2013.ndjson",year=2013, disaster = 'Transportation',location='spain')
df46 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/wildfire-australia-2013.ndjson",year=2013, disaster = 'Wildfire',location='australia')
df47 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/wildfire-california-2014.ndjson",year=2014, disaster = 'Wildfire',location='california')
df48 = new_data("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/incident-tweets/wildfire-colorado-2012.ndjson",year=2011, disaster = 'Wildfire', location='colorado')   

frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
        df11,df12,df13,df14,df15,df16,df17,df17,df18,df19,df20,
        df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,
        df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,
        df41,df42,df43,df44,df45,df46,df47,df48]

df_final = pd.concat(frames)

df = df_final.copy()
def remove_url(text):
    url = re.compile(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*')
    return url.sub(r'', text)

df['text_final'] = df['text'].apply(lambda x: remove_url(x))


#Contractions
df['text_final_1'] = df['text_final'].apply(lambda x: [contractions.fix(word) for word in x.split(' ')])

#joining back the list of items into one string
df['text_final_1'] = [' '.join(map(str, l)) for l in df['text_final_1']]

# Noise Cleaning - spacing, special characters, lowercasing 

df['text_final_1'] = df['text_final_1'].str.lower()
df['text_final_1'] = df['text_final_1'].apply(lambda x: re.sub(r'[^\w\d\s\']+', '', x))
df['text_final_2'] = df['text_final_1'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

#nltk tokenization

df['text_final_1'] = df['text_final_1'].apply(word_tokenize)
df['text_final_2'] = df['text_final_2'].apply(word_tokenize)

# remove stop words

stop_words = set(stopwords.words('english'))

df['text_final_1'] = df['text_final_1'].apply(lambda x: [word for word in x if word not in stop_words])
df['text_final_1'] = [' '.join(map(str, l)) for l in df['text_final_1']]

df['text_final_2'] = df['text_final_2'].apply(lambda x: [word for word in x if word not in stop_words])
df['text_final_2'] = [' '.join(map(str, l)) for l in df['text_final_2']]


# lemmatization

lemma = nltk.WordNetLemmatizer()

df['text_final_2'] = df['text_final_2'].apply(lambda x: [lemma.lemmatize(word) for word in x ])
df['text_final_2'] = [''.join(map(str, l)) for l in df['text_final_2']]


col1, col2, col3, col4 = st.columns([5, 5, 10, 5])

with col1:
    st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/dimbat.png", width=200)
with col4:
    st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/NLpower.png", width=200)

with col3:
    st.title("Dashboard Dimbat")



type_col, year_col, year_all, location_col = st.columns([5, 5, 2, 5])
with year_col:
    year_choice = st.slider(
    "choose year",
    min_value=2012,
    max_value=2017,
    step=1,
    value=2014,
    )
with year_all:
    year_all_choice = st.checkbox("all years?")
    
    
        
with type_col:
    type_choice = st.selectbox("Select the type of disaster",('All', 'Biologocal', 'Earthquake', 'Flood', 'Hurricane',
       'Industrial', 'Meteor', 'Haze', 'Societal', 'Tornado',
       'Transportation', 'Wildfire'))
    
with location_col:
    location_choice = st.selectbox("choose location", ('all','bohol', 'california', 'chile', 'costarica',
       'guatemala', 'iran', 'italy', 'mexico', 'nepal', 'pakistan',
       'alberta', 'colorado', 'india', 'manila', 'philipinnes',
       'queensland', 'sardinia', 'srilanka', 'hagupit', 'harvey', 'iram',
       'maria', 'pablo', 'pam', 'sandy', 'yolanda', 'odile', 'savar',
       'texas', 'venezula', 'russia', 'singapore', 'boston', 'brazil',
       'los angeles', 'joplin', 'oklahoma', 'glasgow', 'newyork', 'spain',
       'australia' ))

submitted = st.button('Submit')

filtered_df=df[df.relevance ==1]

if location_choice != "all":
    filtered_df = filtered_df[filtered_df.location == location_choice]
# -- Apply the year filter given by the user
filtered_df = filtered_df[(df.year == year_choice)]
if year_all_choice:
    filtered_df = filtered_df[(df.year == All)]


# -- Apply the type  filter
if type_choice != "All":
    filtered_df = filtered_df[filtered_df.disaster == type_choice]

if submitted:
 
    line_fig = px.histogram(filtered_df,
                       x='year',
                       color='location',
                       title=f'tweets for {type_choice} in Location:{location_choice} ')
    st.plotly_chart(line_fig)
  
