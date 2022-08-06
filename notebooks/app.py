import streamlit as st
import pandas as pd
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('DIMBAT')


@st.cache(allow_output_mutation=True)
def read_file():
    df = pd.read_pickle('../data/data_model.pkl')
    grouper = df.groupby(['predict'])
    return df, grouper

df, grouper = read_file()

topic = st.selectbox('select topic',('Hurricane & Tornado', 'Transportation', 'Flood', 'Industrial',
                                    'Earthquake', 'Societal', 'Wildfire', 'Biological', 'Meteor'))

col1, col2 = st.columns(2, gap="medium")

if topic:
    content = str(list(df[df['predict'] == topic]['text_final_2']))
    wordcloud = WordCloud().generate(content)

    col1.title("Word Cloud")
    fig = plt.figure()
    # fig.set_figwidth(14)
    # fig.set_figheight()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(topic, fontdict=dict(size=9))
    col1.pyplot(fig)

    data = grouper.get_group(topic).groupby('year').count().reset_index()
    data.rename(columns={"score": "tweets"}, inplace=True)
    fig = px.bar(data, x = 'year',y = 'tweets')

    # fig = px.scatter(data, x = 'year',y = 'tweets', size='tweets',size_max=35)
    fig.update_layout(title=topic)

    col2.title("No.of Tweets for a Disaster")
    col2.plotly_chart(fig)

