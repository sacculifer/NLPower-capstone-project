import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(
    page_title="NLPower -- DIsaster MAnagement BAsed on Twitter (DiMBat)",
    page_icon="✅",
    layout="wide",
)

# read csv from a github repo
dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# read csv from a URL
@st.experimental_memo
def get_data(data) -> pd.DataFrame:
    return pd.read_csv(data)

df = get_data('df_final.csv')
df2 = get_data('df2.csv')

df = df.drop(columns=['Unnamed: 0'])
df2 = df2.drop(columns=['Unnamed: 0'])

# dashboard title
st.markdown("<h1 style='text-align: center;'>NLPower</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Dashboard: DIsaster MAnagement BAsed on Twitter (DiMBat)</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Demonstrator:</h2>", unsafe_allow_html=True)

# model try-out
text_in = st.text_input('Write something, I will classify it for you. May take a couple of seconds though ...', key='text')
def clear_text():
    st.session_state["text"] = ""
    
st.button("IMPORTANT: Click here for clearing text input before continuing below", on_click=clear_text)

if text_in:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
    from geopy.geocoders import Nominatim

    #Classification (Disaster type)
    tokenizer1 = AutoTokenizer.from_pretrained("sacculifer/dimbat_disaster_type_distilbert")
    model1 = AutoModelForSequenceClassification.from_pretrained("sacculifer/dimbat_disaster_type_distilbert", from_tf=True)
    classifier = pipeline("text-classification", tokenizer=tokenizer1, model=model1)
    classification = classifier(text_in)

    #NER Location
    tokenizer2 = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
    model2 = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
    loc = pipeline("ner", model=model2, tokenizer=tokenizer2, aggregation_strategy='average')
    location = loc(text_in)

    #Sentiment
    tokenizer3 = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model3 = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    sent = pipeline("text-classification", tokenizer=tokenizer3, model=model3)
    sentiment = sent(text_in)

    label_mapping = {'LABEL_1': 'Disease', 'LABEL_2': 'Earthquake', 'LABEL_3': 'Flood', 'LABEL_4': 'Hurricane & Tornado', 'LABEL_5': 'Wildfire', 'LABEL_6': 'Industrial Accident', 'LABEL_7': 'Societal Crime', 'LABEL_8': 'Transportation Accident', 'LABEL_9': 'Meteor Crash', 'LABEL_0': 'Haze'}
    label = label_mapping[classification[0]['label']]

    st.write(f'This seems to be a disaster -- I think this refers to a disaster of type: {label}')
    st.write(f"The sentiment I detect is {sentiment[0]['label']}.")

    x = []
    if len(location) != 0:
        for j in range(len(location)):
            if location[j]['entity_group'] == 'LOC':
                x.append(location[j]['word'])

    if len(x) == 0:
        st.write('I have not detected any location')
            
    elif len(x) == 1:
        geolocator = Nominatim(user_agent="MyApp")
        entity = geolocator.geocode(str(x))
            
        if entity is not None:

            long = entity.longitude
            lat = entity.latitude

            locreverse = geolocator.reverse(str(entity.latitude) + " , " + str(entity.longitude), language = 'en')
            address = locreverse.raw['address']
            city = address.get('city', '')
            state = address.get('state', '')
            country = address.get('country', '')
            
            st.write(f"I have detected the following location: {x[0]}. The longitude is: {long} -- and the latitude is: {lat}. The country is {country}.")

        else:
            st.write("I detected a location, but I can't find out anything else about it.")

    elif len(x) > 1:
        geolocator = Nominatim(user_agent="MyApp")
        entity = geolocator.geocode(str(x[0]))

        if entity is not None:

            long = entity.longitude
            lat = entity.latitude

            locreverse = geolocator.reverse(str(entity.latitude) + " , " + str(entity.longitude), language = 'en')
            address = locreverse.raw['address']
            city = address.get('city', '')
            state = address.get('state', '')
            country = address.get('country', '')
            st.write(f"I have detected several locations and will give you some info on the first one you typed in, which was: {x[0]}. The longitude is: {long} -- and the latitude is: {lat}. The country is {country}.")
        else:
            st.write("I have detected several locations. I was trying out to find some info one the first location you typed in, but couldn't find out anything.")


st.markdown("<h2 style='text-align: center;'>Data Visualisations and Table</h2>", unsafe_allow_html=True)
# top-level filters
#job_filter = st.selectbox("Select the continent", pd.unique(df2["continent"]))
continent_filter = st.multiselect(
     'Select a continent',
     ['Asia', 'Africa', 'North America', 'South America', 'Oceania', 'Europe'],
     ['Asia', 'Africa', 'North America', 'South America', 'Oceania', 'Europe'])

disaster_filter = st.multiselect(
     'Select a disaster type',
     ['Hurricane & Tornado', 'Transportation', 'Flood', 'Industrial', 'Earthquake', 'Societal', 'Wildfire', 'Biological', 'Meteor'],
     ['Hurricane & Tornado', 'Transportation', 'Flood', 'Industrial', 'Earthquake', 'Societal', 'Wildfire', 'Biological', 'Meteor'])

if len(continent_filter) == 1:
    if continent_filter == ['Asia']:
        b = 'asia'
    if continent_filter == ['Africa']:
        b = 'africa'
    if continent_filter == ['North America']:
        b = 'north america'
    if continent_filter == ['South America']:
        b = 'south america'
    if continent_filter == ['Oceania']:
        b = 'world'
    if continent_filter == ['Europe']:
        b = 'europe'
else:
    b = 'world'

# dataframe filter
df = df[df['continent'].isin(list(continent_filter)) & df['predict'].isin(list(disaster_filter))]


# create dataframe 2
df2 = df[['country', 'alpha3', 'year', 'continent', 'predict']]
df2 = df2.groupby(['country', 'alpha3', 'year', 'continent', 'predict'])['country'].count()
df2 = pd.DataFrame(df2)
df2.rename(columns = {'country':'count'}, inplace=True)
df2.reset_index(inplace=True)

df3 = df[['country', 'alpha3', 'year', 'continent']]
df3 = df3.groupby(['country', 'alpha3', 'year', 'continent'])['country'].count()
df3 = pd.DataFrame(df3)
df3.rename(columns = {'country':'count'}, inplace=True)
df3.reset_index(inplace=True)
df3['predict'] = 'All categories'
df3 = df3[['country', 'alpha3', 'year', 'continent', 'predict', 'count']]

df2 = pd.concat([df2, df3])
df2 = df2.pivot(index=['country', 'alpha3', 'year', 'continent'], columns='predict', values='count')
df2.reset_index(inplace=True)
df2 = df2.rename_axis(None, axis=1)

# creating KPIs
count = df2['continent'].value_counts()[0]

# create three columns
kpi1, kpi2, kpi3 = st.columns(3)

# fill in those three columns with respective metrics or KPIs
kpi1.metric(
    label="No. Tweets (based on your selection)",
    value=len(df),
    #delta=round(df2['continent'].value_counts()[0]) - 10,
    )

# create two columns for charts
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.markdown("<h3 style='text-align: center; '> Map: No./Tweets per Country</h3>", unsafe_allow_html=True)
    fig = px.choropleth(df2
                        ,locationmode='country names'
                        ,locations = 'country'
                        ,scope = b
                        ,color = "All categories"
                        ,animation_frame = "year"
                        ,hover_name="country"
                    )


    fig.update_geos(visible=True, showcountries=True, projection_type='natural earth', lonaxis_showgrid=False, lataxis_showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    #st.write(fig)


with fig_col2:
    st.markdown("<h3 style='text-align: center; '> Geographical Distribution of Tweets</h3>", unsafe_allow_html=True)
    fig2 = px.histogram(df, color = 'continent', y = 'predict')
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("<h3 style='text-align: center; '> No. of Tweets per Disaster Type and Year</h3>", unsafe_allow_html=True)
fig_tab1 = st.empty()
with fig_tab1:
    fig5 = px.scatter(df.groupby(['predict','year'])['text'].count().reset_index(),
                 x = 'year',y = 'predict', color = 'predict', size='text',size_max=35)
    #fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

fig_col3, fig_col4 = st.columns(2)
with fig_col3:
    st.markdown("<h3 style='text-align: center; '> Emotional Classification of Tweets</h3>", unsafe_allow_html=True)
    fig3 = px.histogram(df, x = 'label',color = 'label')
    st.plotly_chart(fig3, use_container_width=True)


content = str(list(df['text_final_2']))
wordcloud = WordCloud().generate(content)

with fig_col4:
    st.markdown("<h3 style='text-align: center; '> Wordclowd (most freqeuent words)</h3>", unsafe_allow_html=True)
    fig4 = plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    #fig_col2.pyplot(fig2)
    #st.plotly_chart(fig2, use_container_width=False)
    st.write(fig4)

st.markdown("<h3 style='text-align: center; '> Tweets</h3>", unsafe_allow_html=True)

fig_tab2 = st.empty()
with fig_tab2:
    df_display = df[['text', 'predict', 'label', 'country']]
    st.dataframe(df_display)