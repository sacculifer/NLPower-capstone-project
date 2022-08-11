import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go

st.set_page_config(
    page_title="NLPower -- DIsaster Management BAsed on Twitter (DiMBat)",
    page_icon="✅",
    layout="wide",
)

# read csv from a URL
@st.experimental_memo
def get_data(data) -> pd.DataFrame:
    return pd.read_csv(data)

df = get_data('df_final.csv')
#df2 = get_data('df2.csv')

df = df.drop(columns=['Unnamed: 0'])
#df2 = df2.drop(columns=['Unnamed: 0'])

# dashboard title

#from PIL import Image
#image1 = Image.open('NLpower.png')
#image2 = Image.open('dimbat.png')
#img1, img2, img3, img4 = st.columns(4)
#with img2:
#    st.image(image1)
#with img3:
#    st.image(image2)

#st.markdown("<h1 style='text-align: center;'>DIsaster Management BAsed on Twitter (DiMBat)</h1>", unsafe_allow_html=True)
#st.text("")
#st.text("")

#st.markdown("<h3 style='text-align: left;'>Demonstrator:</h3>", unsafe_allow_html=True)

# col1, col2, col3= st.columns([2,7,2])

# with col1:
#     st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/dimbat.png", width=150)
# with col2:  
#     st.markdown("## DiMBaT: preventing bad things from getting worse")  
# with col3:  
#     st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/NLpower.png", width=150)

    
#model try-out
text_in = st.text_input('Write something disastrous, I will classify it for you. May take a couple of seconds though ...', key='text')
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

st.text("")
st.text("")
st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color: teal !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h3 style='text-align: left;'>Data Visualisations and Table:</h3>", unsafe_allow_html=True)
col1, col2 = st.columns([8,2])
with col1:
    # top-level filters
    #job_filter = st.selectbox("Select the continent", pd.unique(df2["continent"],color: green))
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


with col2:  
    
    st.text("")
    
    count = df2['continent'].value_counts()[0]
    st.header(" No. Tweets")
    st.metric(label= "(based on your selection)", value=len(df), delta=None, delta_color="normal")






# create two columns for charts
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    #st.markdown("<p style='text-align: left; '> Map: No./Tweets per Country:</p>", unsafe_allow_html=True)
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
    #st.markdown("<p style='text-align: left; '> Geographical Distribution of Tweets:</p>", unsafe_allow_html=True)
    fig2 = px.histogram(df, color = 'continent', y = 'predict')
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("<p style='text-align: left; '> No. of Tweets per Disaster Type and Year:</p>", unsafe_allow_html=True)
fig_tab1 = st.empty()
with fig_tab1:
    fig5 = px.scatter(df.groupby(['predict','year'])['text'].count().reset_index(),
                 x = 'year',y = 'predict', color = 'predict', size='text',size_max=35)
    #fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

fig_col3, fig_col4 = st.columns(2)
with fig_col3:
    #st.markdown("<p style='text-align: left; '> Emotional Classification of Tweets:</p>", unsafe_allow_html=True)
    #fig3 = px.histogram(df, x = 'label',color = 'label')
    emo_df=df.groupby(["label"])['label'].count()
    emo_df = emo_df.rename_axis(index=None).reset_index()
    pd.DataFrame(emo_df)
    fig3 = go.Figure(data=[go.Pie(labels=emo_df['index'], values=emo_df.label,textinfo = "none" ,marker_colors=px.colors.qualitative.Prism, hole=.6)])
    fig3.update_layout(autosize=False,width=350, height=300,margin_autoexpand=True, margin_b=40, margin_l= 50, margin_t=20, annotations=[dict(text='Tweet Sentiments', x=0.50, y=0.5, font_size=12, showarrow=False)])
    st.plotly_chart(fig3, use_container_width=True)


content = str(list(df['text_final_2']))
wordcloud = WordCloud(background_color='white',).generate(content)

with fig_col4:
    #st.markdown("<p style='text-align: center; '> Wordclowd (most freqeuent words):</p>", unsafe_allow_html=True)
    fig4 = plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    #fig_col2.pyplot(fig2)
    #st.plotly_chart(fig2, use_container_width=False)
    st.write(fig4)

st.markdown("<p style='text-align: left; '> Tweets:</p>", unsafe_allow_html=True)

fig_tab2 = st.empty()
with fig_tab2:
    df_display = df[['text', 'predict', 'label', 'country']]
    st.dataframe(df_display)


st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")


col1, col2, col3= st.columns([2,7,2])

with col1:
    st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/dimbat.png", width=150)
with col2:  
    st.markdown("## DIMBAT: preventing bad things from getting worse")  
with col3:  
    st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/NLpower.png", width=150)

st.markdown("<h2 style='text-align: left; color: teal;'>Outlook</h2>", unsafe_allow_html=True)
st.markdown("### - Running (near) real-time on twitter stream")
st.markdown("### - More refined time component (filtering/visualisation)")
st.markdown("### - Include Twitter geoinformation")
st.markdown("### - More themes for classification")
st.markdown("### - More languages")
st.markdown("### - Adding image recognition/matching functionality")
st.markdown("### - Adding other social media services")


st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)