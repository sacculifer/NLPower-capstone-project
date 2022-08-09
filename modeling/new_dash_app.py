from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px  # interactive charts
import streamlit as st 
import plotly.graph_objects as go




df = pd.read_csv("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/output.csv")

st.set_page_config(
    page_title="DIMBAT - all we wont to write about it",
    page_icon=None,
    layout="wide")
col1, col2, col3= st.columns([2,7,2])

with col1:
    st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/dimbat.png", width=150)
with col2:  
    st.markdown("## DIMBAT: preventing bad things from getting worse")  
with col3:  
    st.image("/Users/lenastrokov/neuefische/NLPower-capstone-project/modeling/data/NLpower.png", width=150)

    

topic = st.selectbox('select desaster type',['All','Hurricane & Tornado', 'Transportation', 'Flood', 'Industrial',
       'Earthquake', 'Societal', 'Wildfire', 'Biological', 'Meteor'])

# -- Apply the type  filter
if topic != "All":
    filtered_df = df[df.predict == topic]
    emo_df=filtered_df.groupby("label", as_index=False ).count()
    pred_year_df=filtered_df.groupby(["year","predict"], as_index=False ).count()
    
else:
    filtered_df=df
    emo_df=df.groupby("label", as_index=False ).count()
    pred_year_df=df.groupby(["year","predict"], as_index=False ).count()

fig1 = go.Figure(data=[go.Pie(labels=emo_df.label, values=emo_df.text,textinfo = "none" ,marker_colors=px.colors.qualitative.Prism, hole=.6)])
fig1.update_layout(autosize=False,width=350, height=300,margin_autoexpand=False, margin_b=40, margin_l= 0, margin_t=20,
   
    annotations=[dict(text='tweet emotions', x=0.50, y=0.5, font_size=12, showarrow=False)])

fig2= px.bar(pred_year_df, x="year", y="text",color="predict",labels={'text':'count of related tweets'},color_discrete_sequence=px.colors.qualitative.Set2)
fig2.update_layout(autosize=False,width=600, height=300,margin_autoexpand=False, margin_b=40, margin_l= 80, margin_t=0)
fig2.update_xaxes(type='category')


  
text= str(list(filtered_df["text_cloud"]))
wordcloud = WordCloud(background_color="white",width=350, height=350).generate(text)
fig3, ax = plt.subplots(figsize = (3, 3))
ax.imshow(wordcloud)
plt.axis("off")


fig4 = go.Figure(go.Densitymapbox(lat= filtered_df.lat1, lon=filtered_df.long1,# z=df.score,
                                 radius=2))
fig4.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=-31)
fig4.update_layout(autosize=False,width=600, height=320, margin_autoexpand=False, margin_b=40, margin_l= 120, margin_t=0)
fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


c1,c2 = st.columns([2,3])


with c1:
    #c1.header("A cat")
    st.plotly_chart(fig1)
with c2:
    #c2.header("A dog")
    st.plotly_chart(fig2)


   # st.plotly_chart(fig4)

c3,c4 = st.columns([1,2])
with c3:
    #c1.header("A cat")
    st.pyplot(fig3)
with c4:
    #c2.header("A dog")
    st.plotly_chart(fig4)






      




