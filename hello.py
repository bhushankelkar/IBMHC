from cloudant import Cloudant #!!COMMENTED FROM BHUSHAN'S CODE!!
from flask import Flask, render_template, request, jsonify
import atexit #!!COMMENTED FROM BHUSHAN'S CODE!!
import time
import os
import json
import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html 
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go #added
import pandas as pd
import plotly.io as pio
import plotly.express as px
import nltk
import regex as re
from plotly.subplots import make_subplots
import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer #word stemmer class
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle
import string
import joblib
from nltk.corpus import stopwords

lemma = WordNetLemmatizer()

app = dash.Dash(__name__)
neu=0
pos = 0
neg = 0
time = 0
vectorizer = pickle.load(open("vector.pickel", "rb"))
w = pickle.load(open('train', 'rb'))
x_train=w['x_train']
vectorizer.fit(x_train)
print(vectorizer)
print(len(vectorizer.get_feature_names()))
filename = 'logR.sav'
tf2=pd.DataFrame(columns=['pos','neg','neu','time'])
loaded_model = joblib.load(open(filename, 'rb'))
positive = 0
negative = 0
neutral = 0
compound=0
ps = PorterStemmer()
wnl = WordNetLemmatizer()
cal={'val':['positive','negative','neutral'],'count':[positive,negative,neutral]}
cal=pd.DataFrame(cal)

df1 = pd.read_csv('Lockdowns/lock1.csv',encoding='latin')
df2 = pd.read_csv('Lockdowns/lock2.csv',encoding='latin')
df3 = pd.read_csv('Lockdowns/lock3.csv',encoding='latin')
df4 = pd.read_csv('Lockdowns/lock4.csv',encoding='latin')

df1[['day','time']]=df1.date.str.split(expand=True)
df1=df1.drop(['date'],axis=1)
df2[['day','time']]=df2.date.str.split(expand=True)
df2=df2.drop(['date'],axis=1)
df3[['day','time']]=df3.date.str.split(expand=True)
df3=df3.drop(['date'],axis=1)
df4[['day','time']]=df4.date.str.split(expand=True)
df4=df4.drop(['date'],axis=1)

dates = pd.DataFrame()
dates2 = pd.DataFrame()
dates3 = pd.DataFrame()
dates4 = pd.DataFrame()
dates = df1[['day','time','labels','text']]
date2 = df2[['day','time','labels','text']]
date3 = df3[['day','time','labels','text']]
date4 = df4[['day','time','labels','text']]
dates = dates.append(date2,ignore_index=True)
dates = dates.append(date3,ignore_index=True)
dates = dates.append(date4,ignore_index=True)

lemma = WordNetLemmatizer()

# Use this for hashtag extract
port = int(os.getenv('PORT', 8000))
def hashtag_extract(x):
    hashtags = []

    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        ht=[hts.lower() for hts in ht]
        #ht=map(str.lower,ht)


        hashtags.append(ht)
    return hashtags
#-------------------------Lockdown1------------------
HT_regular = hashtag_extract(df1['text'][df1['labels'] == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(df1['text'][df1['labels'] == -1])
HT_positive = hashtag_extract(df1['text'][df1['labels'] == 1])
# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
HT_positive = sum(HT_positive,[])
#print(HT_regular,file=sys.stderr)
#print(HT_negative,file=sys.stderr)
#positive hashtags
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
d.head()
#plt.figure(figsize=(22,10))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()
#negative hastags funtion will come over here
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 5)   
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()'''
g = nltk.FreqDist(HT_positive)
h = pd.DataFrame({'Hashtag': list(g.keys()),
                  'Count': list(g.values())})
# selecting top 10 most frequent hashtags     
h = h.nlargest(columns="Count", n = 5) 

#-----------------------Lockdown2----------------------------
HT_regular = hashtag_extract(df2['text'][df2['labels'] == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(df2['text'][df2['labels'] == -1])
HT_positive = hashtag_extract(df2['text'][df2['labels'] == 1])
# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
HT_positive = sum(HT_positive,[])
#print(HT_regular,file=sys.stderr)
#print(HT_negative,file=sys.stderr)
#positive hashtags
a2 = nltk.FreqDist(HT_regular)
d2 = pd.DataFrame({'Hashtag': list(a2.keys()),
                  'Count': list(a2.values())})
# selecting top 10 most frequent hashtags     
d2 = d2.nlargest(columns="Count", n = 5) 
d2.head()
#plt.figure(figsize=(22,10))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()
#negative hastags funtion will come over here
b2 = nltk.FreqDist(HT_negative)
e2 = pd.DataFrame({'Hashtag': list(b2.keys()), 'Count': list(b2.values())})
# selecting top 10 most frequent hashtags
e2 = e2.nlargest(columns="Count", n = 5)   
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()'''
g2 = nltk.FreqDist(HT_positive)
h2 = pd.DataFrame({'Hashtag': list(g2.keys()),
                  'Count': list(g2.values())})
# selecting top 10 most frequent hashtags     
h2 = h2.nlargest(columns="Count", n = 5) 

#---------------------Lockdown3---------------------
HT_regular = hashtag_extract(df3['text'][df3['labels'] == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(df3['text'][df3['labels'] == -1])
HT_positive = hashtag_extract(df3['text'][df3['labels'] == 1])
# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
HT_positive = sum(HT_positive,[])
#print(HT_regular,file=sys.stderr)
#print(HT_negative,file=sys.stderr)
#positive hashtags
a3 = nltk.FreqDist(HT_regular)
d3 = pd.DataFrame({'Hashtag': list(a3.keys()),
                  'Count': list(a3.values())})
# selecting top 10 most frequent hashtags     
d3 = d3.nlargest(columns="Count", n = 5) 
d3.head()
#plt.figure(figsize=(22,10))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()
#negative hastags funtion will come over here
b3 = nltk.FreqDist(HT_negative)
e3 = pd.DataFrame({'Hashtag': list(b3.keys()), 'Count': list(b3.values())})
# selecting top 10 most frequent hashtags
e3 = e3.nlargest(columns="Count", n = 5)   
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()'''
g3 = nltk.FreqDist(HT_positive)
h3 = pd.DataFrame({'Hashtag': list(g3.keys()),
                  'Count': list(g3.values())})
# selecting top 10 most frequent hashtags     
h3 = h3.nlargest(columns="Count", n = 5) 

#-------------------------Lockdown4---------------------
HT_regular = hashtag_extract(df4['text'][df4['labels'] == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(df4['text'][df4['labels'] == -1])
HT_positive = hashtag_extract(df4['text'][df4['labels'] == 1])
# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
HT_positive = sum(HT_positive,[])
#print(HT_regular,file=sys.stderr)
#print(HT_negative,file=sys.stderr)
#positive hashtags
a4 = nltk.FreqDist(HT_regular)
d4 = pd.DataFrame({'Hashtag': list(a4.keys()),
                  'Count': list(a4.values())})
# selecting top 10 most frequent hashtags     
d4 = d4.nlargest(columns="Count", n = 5) 
d4.head()
#plt.figure(figsize=(22,10))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()
#negative hastags funtion will come over here
b4 = nltk.FreqDist(HT_negative)
e4 = pd.DataFrame({'Hashtag': list(b4.keys()), 'Count': list(b4.values())})
# selecting top 10 most frequent hashtags
e4 = e4.nlargest(columns="Count", n = 5)   
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()'''
g4 = nltk.FreqDist(HT_positive)
h4 = pd.DataFrame({'Hashtag': list(g4.keys()),
                  'Count': list(g4.values())})
# selecting top 10 most frequent hashtags     
h4 = h4.nlargest(columns="Count", n = 5) 

#Use this for wordcount
'''
def word_count(sentence):
    return len(sentence.split())
df['word count'] = df['text'].apply(word_count)
x = df['word count'][df.labels == 1]
y = df['word count'][df.labels == 0]
print(x,file=sys.stderr)
print(y,file=sys.stderr)
#plt.figure(figsize=(12,6))
#plt.xlim(0,45)
#plt.xlabel('word count')
#plt.ylabel('frequency')
#g = plt.hist([x, y], color=['r','b'], alpha=0.5, label=['positive','negative'])
#plt.legend(loc='upper right')
#Till here'''


app.layout =  html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
    dcc.Tab(label='Live Tweets', value='tab-1',children=[dcc.Interval(
        id='interval-component-slow',
        interval=1*1000,
        n_intervals=0  # in milliseconds
        )]),
        dcc.Tab(label='Lockdown 1.0', value='tab-2'),
        dcc.Tab(label='Lockdown 2.0', value='tab-3'),
        dcc.Tab(label='Lockdown 3.0', value='tab-4'),
        dcc.Tab(label='Lockdown 4.0', value='tab-5'),
        dcc.Tab(label='Lockdown Analysis', value='tab-6',children = [dcc.Dropdown(
        id='date-dropdown',
        options=[{'label': i, 'value': i} for i in dates['day'].unique()],
        value='25-03-2020',
        multi = False,
        style = {'width':'40%'}

    )]),
    ],
      colors={
                "border":"#eeeeee",
                "primary":"#679b9b",
                "background":"#2e9cc8",

      }
    ),
    html.Div(id='tabs-content')
])
#------------------------------------------DONUT CHART----------------------------------------------------------
count1=(df1['labels'][df1['labels']==1]).count()
count1loc2=(df2['labels'][df2['labels']==1]).count()
count1loc3=(df3['labels'][df3['labels']==1]).count()
count1loc4=(df4['labels'][df4['labels']==1]).count()
countneg=(df1['labels'][df1['labels']==-1]).count()
countneg2=(df2['labels'][df2['labels']==-1]).count()
countneg3=(df3['labels'][df3['labels']==-1]).count()
countneg4=(df4['labels'][df4['labels']==-1]).count()
count0=(df1['labels'][df1['labels']==0]).count()
count0loc2=(df2['labels'][df2['labels']==0]).count()
count0loc3=(df3['labels'][df3['labels']==0]).count()
count0loc4=(df4['labels'][df4['labels']==0]).count()

print(count1)
colors=['#ff9595','royalblue','#80bdab']
#--------------------------------------HASHTAGS SUBPLOTS---------------------------------------------------------
#-----------------Lockdown1--------------------------------
fig = make_subplots(rows=1, cols=3)

fig.add_trace(
    go.Bar(y=e.Hashtag,
                x=e.Count,
                name='# Negative',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(y=h.Hashtag,
                x=h.Count,
                name='# Positive',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=2
)

fig.add_trace(
    go.Bar(y=d.Hashtag,
                x=d.Count,
                name='# Neutral',
                textfont_color='white',

                orientation='h'),
    row=1, col=3
)


fig.update_layout( 
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Popular Hashtags"

                )

#----------------Lockdown2----------------------
figloc2 = make_subplots(rows=1, cols=3)

figloc2.add_trace(
    go.Bar(y=e2.Hashtag,
                x=e2.Count,
                name='# Negative',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=1
)

figloc2.add_trace(
    go.Bar(y=h2.Hashtag,
                x=h2.Count,
                name='# Positive',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=2
)

figloc2.add_trace(
    go.Bar(y=d2.Hashtag,
                x=d2.Count,
                name='# Neutral',
                textfont_color='white',

                orientation='h'),
    row=1, col=3
)



figloc2.update_layout( 
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Popular Hashtags"

                )

#--------------------Lockdown3-------------------
figloc3 = make_subplots(rows=1, cols=3)

figloc3.add_trace(
    go.Bar(y=e3.Hashtag,
                x=e3.Count,
                name='# Negative',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=1
)

figloc3.add_trace(
    go.Bar(y=h3.Hashtag,
                x=h3.Count,
                name='# Positive',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=2
)

figloc3.add_trace(
    go.Bar(y=d3.Hashtag,
                x=d3.Count,
                name='# Neutral',
                textfont_color='white',

                orientation='h'),
    row=1, col=3
)



figloc3.update_layout( 
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Popular Hashtags"

                )

#-------------Lockdown4-----------------------
figloc4 = make_subplots(rows=1, cols=3)

figloc4.add_trace(
    go.Bar(y=e4.Hashtag,
                x=e4.Count,
                name='# Negative',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=1
)

figloc4.add_trace(
    go.Bar(y=h4.Hashtag,
                x=h4.Count,
                name='# Positive',
                textfont_color='white',
                
                orientation='h'),
    row=1, col=2
)

figloc4.add_trace(
    go.Bar(y=d4.Hashtag,
                x=d4.Count,
                name='# Neutral',
                textfont_color='white',

                orientation='h'),
    row=1, col=3
)



figloc4.update_layout( 
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Popular Hashtags"

                )


#----------------------------------------WATSON TONE ANALYSER------------------------------------------------------------
tone=pd.read_csv('Tone Analyser/Lock1ToneAnalyser.csv')
count_sad=tone['sadness'].sum()
count_joy=tone['joy'].sum()
count_confident=tone['confident'].sum()
count_analytical=tone['analytical'].sum()
count_tentative=tone['tentative'].sum()
count_fear=tone['fear'].sum()
count_anger=tone['anger'].sum()

colors_emo=['#ff9595','#ff9595','#ff9595','#80bdab','royalblue','royalblue','royalblue']

fig1 = go.Figure([go.Bar(
             x=['😁','😎','🧐','😐','🥺','😨','😡'],
             y=[count_joy,count_confident,count_analytical,count_tentative,count_sad,count_fear,count_anger],
             hovertext=['Happy','Hopeful','Analytical','Neutral','Sad','Fearful','Angry'],
             hoverinfo='text+y',
             marker_color=colors_emo
             

    )])




fig1.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Tone Analysis",
                #yaxis_tickformat='percent'
                )
fig1.update_xaxes(tickfont=dict(size=25))

#--------------------Lockdown2----------------------
tone=pd.read_csv('Tone Analyser/Lock2ToneAnalyser.csv')
count_sad=tone['sadness'].sum()
count_joy=tone['joy'].sum()
count_confident=tone['confident'].sum()
count_analytical=tone['analytical'].sum()
count_tentative=tone['tentative'].sum()
count_fear=tone['fear'].sum()
count_anger=tone['anger'].sum()

colors_emo=['#ff9595','#ff9595','#ff9595','#80bdab','royalblue','royalblue','royalblue']

fig1loc2 = go.Figure([go.Bar(
             x=['😁','😎','🧐','😐','🥺','😨','😡'],
             y=[count_joy,count_confident,count_analytical,count_tentative,count_sad,count_fear,count_anger],
             hovertext=['Happy','Hopeful','Analytical','Neutral','Sad','Fearful','Angry'],
             hoverinfo='text+y',
             marker_color=colors_emo
             

    )])




fig1loc2.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Tone Analysis",
                #yaxis_tickformat='percent'
                )
fig1loc2.update_xaxes(tickfont=dict(size=25))

#--------------------Lockdown3----------------------
tone=pd.read_csv('Tone Analyser/Lock3ToneAnalyser.csv')
count_sad=tone['sadness'].sum()
count_joy=tone['joy'].sum()
count_confident=tone['confident'].sum()
count_analytical=tone['analytical'].sum()
count_tentative=tone['tentative'].sum()
count_fear=tone['fear'].sum()
count_anger=tone['anger'].sum()

colors_emo=['#ff9595','#ff9595','#ff9595','#80bdab','royalblue','royalblue','royalblue']

fig1loc3 = go.Figure([go.Bar(
             x=['😁','😎','🧐','😐','🥺','😨','😡'],
             y=[count_joy,count_confident,count_analytical,count_tentative,count_sad,count_fear,count_anger],
             hovertext=['Happy','Hopeful','Analytical','Neutral','Sad','Fearful','Angry'],
             hoverinfo='text+y',
             marker_color=colors_emo
             

    )])




fig1loc3.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Tone Analysis",
                #yaxis_tickformat='percent'
                )
fig1loc3.update_xaxes(tickfont=dict(size=25))

#----------------Lockdown4-------------------
tone=pd.read_csv('Tone Analyser/Lock4ToneAnalyser.csv')
count_sad=tone['sadness'].sum()
count_joy=tone['joy'].sum()
count_confident=tone['confident'].sum()
count_analytical=tone['analytical'].sum()
count_tentative=tone['tentative'].sum()
count_fear=tone['fear'].sum()
count_anger=tone['anger'].sum()

colors_emo=['#ff9595','#ff9595','#ff9595','#80bdab','royalblue','royalblue','royalblue']

fig1loc4 = go.Figure([go.Bar(
             x=['😁','😎','🧐','😐','🥺','😨','😡'],
             y=[count_joy,count_confident,count_analytical,count_tentative,count_sad,count_fear,count_anger],
             hovertext=['Happy','Hopeful','Analytical','Neutral','Sad','Fearful','Angry'],
             hoverinfo='text+y',
             marker_color=colors_emo
             

    )])




fig1loc4.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Tone Analysis",
                #yaxis_tickformat='percent'
                )
fig1loc4.update_xaxes(tickfont=dict(size=25))


#-------------------------------------------------LINE CHART-------------------------------------------------------------
#-----------------Lockdown1-------------------------
df_p1=pd.read_csv('Line Graph/Lock1/df_p1.csv')
df_neg1=pd.read_csv('Line Graph/Lock1/df_neg1.csv')
df_neu1=pd.read_csv('Line Graph/Lock1/df_neu1.csv')
fig2=go.Figure()
    
#fig2=tools.make_subplots(rows=1,cols=3,shared_xaxes=True,shared_yaxes=True)
fig2.add_trace(go.Scatter(x=df_neg1.day,
    y=df_neg1.neg,name='negatives'))
fig2.add_trace(go.Scatter(x=df_p1['day'],
    y=df_p1['pos'],name='positives'))

fig2.add_trace(go.Scatter( x=df_neu1.day,
    y=df_neu1.neu,name='neutrals'))

fig2['layout'].update( title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Emotional Trends in Lockdown 1.0",
                #yaxis_tickformat='percent'
                )

#-----------Lockdown2----------------
df_p2=pd.read_csv('Line Graph/Lock2/df_p2.csv')
df_neg2=pd.read_csv('Line Graph/Lock2/df_neg2.csv')
df_neu2=pd.read_csv('Line Graph/Lock2/df_neu2.csv')
fig2loc2=go.Figure()
    
#fig2=tools.make_subplots(rows=1,cols=3,shared_xaxes=True,shared_yaxes=True)
fig2loc2.add_trace(go.Scatter(x=df_neg2.day,
    y=df_neg2.neg,name='negatives'))
fig2loc2.add_trace(go.Scatter(x=df_p2['day'],
    y=df_p2['pos'],name='positives'))

fig2loc2.add_trace(go.Scatter( x=df_neu2.day,
    y=df_neu2.neu,name='neutrals'))

fig2loc2['layout'].update( title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Emotional Trends in Lockdown 1.0",
                #yaxis_tickformat='percent'
                )

#----------------Lockdown3------------------------
df_p3=pd.read_csv('Line Graph/Lock3/df_p3.csv')
df_neg3=pd.read_csv('Line Graph/Lock3/df_neg3.csv')
df_neu3=pd.read_csv('Line Graph/Lock3/df_neu3.csv')
fig2loc3=go.Figure()
    
#fig2=tools.make_subplots(rows=1,cols=3,shared_xaxes=True,shared_yaxes=True)
fig2loc3.add_trace(go.Scatter(x=df_neg3.day,
    y=df_neg3.neg,name='negatives'))
fig2loc3.add_trace(go.Scatter(x=df_p3['day'],
    y=df_p3['pos'],name='positives'))

fig2loc3.add_trace(go.Scatter( x=df_neu3.day,
    y=df_neu3.neu,name='neutrals'))

fig2loc3['layout'].update( title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Emotional Trends in Lockdown 1.0",
                #yaxis_tickformat='percent'
                )

#------------Lockdown4-----------
df_p4=pd.read_csv('Line Graph/Lock4/df_p4.csv')
df_neg4=pd.read_csv('Line Graph/Lock4/df_neg4.csv')
df_neu4=pd.read_csv('Line Graph/Lock4/df_neu4.csv')
fig2loc4=go.Figure()
    
#fig2=tools.make_subplots(rows=1,cols=3,shared_xaxes=True,shared_yaxes=True)
fig2loc4.add_trace(go.Scatter(x=df_neg4.day,
    y=df_neg4.neg,name='negatives'))
fig2loc4.add_trace(go.Scatter(x=df_p4['day'],
    y=df_p4['pos'],name='positives'))

fig2loc4.add_trace(go.Scatter( x=df_neu4.day,
    y=df_neu4.neu,name='neutrals'))

fig2loc4['layout'].update( title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Emotional Trends in Lockdown 1.0",
                #yaxis_tickformat='percent'
                )

#fig2.update_xaxes(tickfont=dict(size=5))
#-----------------------------------------------------------------------------------------------------------
#----------------------------------------------THANKFUL-------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/Thank You/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/Thank You/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/Thank You/Lock3.csv')
d4=pd.read_csv('Hashtag Analysis/Thank You/Lock4.csv')
ty_lock1_pos=(d1['labels'][d1['labels']==1]).count()
ty_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
ty_lock1_neu=(d1['labels'][d1['labels']==0]).count()
ty_lock2_pos=(d2['labels'][d2['labels']==1]).count()
ty_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
ty_lock2_neu=(d2['labels'][d2['labels']==0]).count()
ty_lock3_pos=(d3['labels'][d3['labels']==1]).count()
ty_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
ty_lock3_neu=(d3['labels'][d3['labels']==0]).count()
ty_lock4_pos=(d4['labels'][d4['labels']==1]).count()
ty_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
ty_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']

neglock1 = (ty_lock1_neg)/(ty_lock1_pos+ty_lock1_neg+ty_lock1_neu)
neglock2 = (ty_lock2_neg)/(ty_lock2_pos+ty_lock2_neg+ty_lock2_neu)
neglock3 = (ty_lock3_neg)/(ty_lock3_pos+ty_lock3_neg+ty_lock3_neu)
neglock4 = (ty_lock4_neg)/(ty_lock3_pos+ty_lock4_neg+ty_lock4_neu)
 
poslock1 = (ty_lock1_pos)/(ty_lock1_pos+ty_lock1_neg+ty_lock1_neu)
poslock2 = (ty_lock2_pos)/(ty_lock2_pos+ty_lock2_neg+ty_lock2_neu)
poslock3 = (ty_lock3_pos)/(ty_lock3_pos+ty_lock3_neg+ty_lock3_neu)
poslock4 = (ty_lock4_pos)/(ty_lock3_pos+ty_lock4_neg+ty_lock4_neu)

neulock1 = (ty_lock1_neu)/(ty_lock1_pos+ty_lock1_neg+ty_lock1_neu)
neulock2 = (ty_lock2_neu)/(ty_lock2_pos+ty_lock2_neg+ty_lock2_neu)
neulock3 = (ty_lock3_neu)/(ty_lock3_pos+ty_lock3_neg+ty_lock3_neu)
neulock4 = (ty_lock4_neu)/(ty_lock3_pos+ty_lock4_neg+ty_lock4_neu)

print(neglock1+poslock1+neulock1)

figty = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[neglock1,neglock2,neglock3,neglock4],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[poslock1,poslock2,poslock3,poslock4],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[neulock1,neulock2,neulock3,neulock4],textposition='auto')

])
# Change the bar mode
figty.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Reaction of people to events of positivity",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1],
                )
figty.update_yaxes(showticklabels=True)
#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------NEGATIVE-------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/Negative/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/Negative/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/Negative/Lock3.csv')
d4=pd.read_csv('Hashtag Analysis/Negative/Lock4.csv')

nega_lock1_pos=(d1['labels'][d1['labels']==1]).count()
nega_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
nega_lock1_neu=(d1['labels'][d1['labels']==0]).count()
nega_lock2_pos=(d2['labels'][d2['labels']==1]).count()
nega_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
nega_lock2_neu=(d2['labels'][d2['labels']==0]).count()
nega_lock3_pos=(d3['labels'][d3['labels']==1]).count()
nega_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
nega_lock3_neu=(d3['labels'][d3['labels']==0]).count()
nega_lock4_pos=(d4['labels'][d4['labels']==1]).count()
nega_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
nega_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']


fignega = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[(nega_lock1_neg)/(nega_lock1_pos+nega_lock1_neg+nega_lock1_neu)
,(nega_lock2_neg)/(nega_lock2_pos+nega_lock2_neg+nega_lock2_neu)
,(nega_lock3_neg)/(nega_lock3_pos+nega_lock3_neg+nega_lock3_neu)
,(nega_lock4_neg)/(nega_lock4_pos+nega_lock4_neg+nega_lock4_neu)
],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[(nega_lock1_pos)/(nega_lock1_pos+nega_lock1_neg+nega_lock1_neu)
,(nega_lock2_pos)/(nega_lock2_pos+nega_lock2_neg+nega_lock2_neu)
,(nega_lock3_pos)/(nega_lock3_pos+nega_lock3_neg+nega_lock3_neu)
,(nega_lock4_pos)/(nega_lock4_pos+nega_lock4_neg+nega_lock4_neu)
],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[(nega_lock1_neu)/(nega_lock1_pos+nega_lock1_neg+nega_lock1_neu)
,(nega_lock2_neu)/(nega_lock2_pos+nega_lock2_neg+nega_lock2_neu)
,(nega_lock3_neu)/(nega_lock3_pos+nega_lock3_neg+nega_lock3_neu)
,(nega_lock4_neu)/(nega_lock4_pos+nega_lock4_neg+nega_lock4_neu)
],textposition='auto')

])
# Change the bar mode
fignega.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Reaction of the people towards negative events",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1]


    )
fignega.update_yaxes(showticklabels=True)
#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------WFH-------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/WFH/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/WFH/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/WFH/Lock2.csv')
d4=pd.read_csv('Hashtag Analysis/WFH/Lock2.csv')
wfh_lock1_pos=(d1['labels'][d1['labels']==1]).count()
wfh_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
wfh_lock1_neu=(d1['labels'][d1['labels']==0]).count()
wfh_lock2_pos=(d2['labels'][d2['labels']==1]).count()
wfh_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
wfh_lock2_neu=(d2['labels'][d2['labels']==0]).count()
wfh_lock3_pos=(d3['labels'][d3['labels']==1]).count()
wfh_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
wfh_lock3_neu=(d3['labels'][d3['labels']==0]).count()
wfh_lock4_pos=(d4['labels'][d4['labels']==1]).count()
wfh_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
wfh_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']

figwfh = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[(wfh_lock1_neg)/(wfh_lock1_pos+wfh_lock1_neg+wfh_lock1_neu),(wfh_lock2_neg)/(wfh_lock2_pos+wfh_lock2_neg+wfh_lock2_neu),(wfh_lock3_neg)/(wfh_lock3_pos+wfh_lock3_neg+wfh_lock3_neu),(wfh_lock4_neg)/(wfh_lock4_pos+wfh_lock4_neg+wfh_lock4_neu)],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[(wfh_lock1_pos)/(wfh_lock1_pos+wfh_lock1_neg+wfh_lock1_neu),(wfh_lock2_pos)/(wfh_lock2_pos+wfh_lock2_neg+wfh_lock2_neu),(wfh_lock3_pos)/(wfh_lock3_pos+wfh_lock3_neg+wfh_lock3_neu),(wfh_lock4_pos)/(wfh_lock4_pos+wfh_lock4_neg+wfh_lock4_neu)],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[(wfh_lock1_neu)/(wfh_lock1_pos+wfh_lock1_neg+wfh_lock1_neu),(wfh_lock2_neu)/(wfh_lock2_pos+wfh_lock2_neg+wfh_lock2_neu),(wfh_lock3_neu)/(wfh_lock1_pos+wfh_lock1_neg+wfh_lock1_neu),(wfh_lock4_neu)/(wfh_lock4_pos+wfh_lock4_neg+wfh_lock4_neu)],textposition='auto')
])
# Change the bar mode
figwfh.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Work from home? Urghhh... OR Yayyy",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1]


    )
figwfh.update_yaxes(showticklabels=True)
#--------------------------------------------------------------------------------------------------------------
#----------------------------------------------Mental Health-------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/Mental Health/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/Mental Health/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/Mental Health/Lock3.csv')
d4=pd.read_csv('Hashtag Analysis/Mental Health/Lock4.csv')

mh_lock1_pos=(d1['labels'][d1['labels']==1]).count()
mh_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
mh_lock1_neu=(d1['labels'][d1['labels']==0]).count()
mh_lock2_pos=(d2['labels'][d2['labels']==1]).count()
mh_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
mh_lock2_neu=(d2['labels'][d2['labels']==0]).count()
mh_lock3_pos=(d3['labels'][d3['labels']==1]).count()
mh_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
mh_lock3_neu=(d3['labels'][d3['labels']==0]).count()
mh_lock4_pos=(d4['labels'][d4['labels']==1]).count()
mh_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
mh_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']

figmh = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[(mh_lock1_neg)/(mh_lock1_pos+mh_lock1_neg+mh_lock1_neu),(mh_lock2_neg)/(mh_lock2_pos+mh_lock2_neg+mh_lock2_neu),(mh_lock3_neg)/(mh_lock3_pos+mh_lock3_neg+mh_lock3_neu),(mh_lock4_neg)/(mh_lock4_pos+mh_lock4_neg+mh_lock4_neu)],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[(mh_lock1_pos)/(mh_lock1_pos+mh_lock1_neg+mh_lock1_neu),(mh_lock2_pos)/(mh_lock2_pos+mh_lock2_neg+mh_lock2_neu),(mh_lock3_pos)/(mh_lock3_pos+mh_lock3_neg+mh_lock3_neu),(mh_lock4_pos)/(mh_lock4_pos+mh_lock4_neg+mh_lock4_neu)],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[(mh_lock1_neu)/(mh_lock1_pos+mh_lock1_neg+mh_lock1_neu),(mh_lock2_neu)/(mh_lock2_pos+mh_lock2_neg+mh_lock2_neu),(mh_lock3_neu)/(mh_lock1_pos+mh_lock1_neg+mh_lock1_neu),(mh_lock4_neu)/(mh_lock4_pos+mh_lock4_neg+mh_lock4_neu)],textposition='auto')
])
# Change the bar mode
figmh.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Mental health",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1]


    )
figmh.update_yaxes(showticklabels=True)

#--------------------------------------------------------------------------------------------------------------
#----------------------------------------------Economy------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/Economy/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/Economy/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/Economy/Lock3.csv')
d4=pd.read_csv('Hashtag Analysis/Economy/Lock4.csv')

eco_lock1_pos=(d1['labels'][d1['labels']==1]).count()
eco_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
eco_lock1_neu=(d1['labels'][d1['labels']==0]).count()
eco_lock2_pos=(d2['labels'][d2['labels']==1]).count()
eco_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
eco_lock2_neu=(d2['labels'][d2['labels']==0]).count()
eco_lock3_pos=(d3['labels'][d3['labels']==1]).count()
eco_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
eco_lock3_neu=(d3['labels'][d3['labels']==0]).count()
eco_lock4_pos=(d4['labels'][d4['labels']==1]).count()
eco_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
eco_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']

figeco = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[(eco_lock1_neg)/(eco_lock1_pos+eco_lock1_neg+eco_lock1_neu),(eco_lock2_neg)/(eco_lock2_pos+eco_lock2_neg+eco_lock2_neu),(eco_lock3_neg)/(eco_lock3_pos+eco_lock3_neg+eco_lock3_neu),(eco_lock4_neg)/(eco_lock4_pos+eco_lock4_neg+eco_lock4_neu)],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[(eco_lock1_pos)/(eco_lock1_pos+eco_lock1_neg+eco_lock1_neu),(eco_lock2_pos)/(eco_lock2_pos+eco_lock2_neg+eco_lock2_neu),(eco_lock3_pos)/(eco_lock3_pos+eco_lock3_neg+eco_lock3_neu),(eco_lock4_pos)/(eco_lock4_pos+eco_lock4_neg+eco_lock4_neu)],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[(eco_lock1_neu)/(eco_lock1_pos+eco_lock1_neg+eco_lock1_neu),(eco_lock2_neu)/(eco_lock2_pos+eco_lock2_neg+eco_lock2_neu),(eco_lock3_neu)/(eco_lock1_pos+eco_lock1_neg+eco_lock1_neu),(eco_lock4_neu)/(eco_lock4_pos+eco_lock4_neg+eco_lock4_neu)],textposition='auto')
])
# Change the bar mode
figeco.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="How about Economy?",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1]


    )
figeco.update_yaxes(showticklabels=True)
#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------Govt------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/Govt/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/Govt/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/Govt/Lock3.csv')
d4=pd.read_csv('Hashtag Analysis/Govt/Lock4.csv')

govt_lock1_pos=(d1['labels'][d1['labels']==1]).count()
govt_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
govt_lock1_neu=(d1['labels'][d1['labels']==0]).count()
govt_lock2_pos=(d2['labels'][d2['labels']==1]).count()
govt_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
govt_lock2_neu=(d2['labels'][d2['labels']==0]).count()
govt_lock3_pos=(d3['labels'][d3['labels']==1]).count()
govt_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
govt_lock3_neu=(d3['labels'][d3['labels']==0]).count()
govt_lock4_pos=(d4['labels'][d4['labels']==1]).count()
govt_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
govt_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']

figgovt = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[(govt_lock1_neg)/(govt_lock1_pos+govt_lock1_neg+govt_lock1_neu),(govt_lock2_neg)/(govt_lock2_pos+govt_lock2_neg+govt_lock2_neu),(govt_lock3_neg)/(govt_lock3_pos+govt_lock3_neg+govt_lock3_neu),(govt_lock4_neg)/(govt_lock4_pos+govt_lock4_neg+govt_lock4_neu)],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[(govt_lock1_pos)/(govt_lock1_pos+govt_lock1_neg+govt_lock1_neu),(govt_lock2_pos)/(govt_lock2_pos+govt_lock2_neg+govt_lock2_neu),(govt_lock3_pos)/(govt_lock3_pos+govt_lock3_neg+govt_lock3_neu),(govt_lock4_pos)/(govt_lock4_pos+govt_lock4_neg+govt_lock4_neu)],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[(govt_lock1_neu)/(govt_lock1_pos+govt_lock1_neg+govt_lock1_neu),(govt_lock2_neu)/(govt_lock2_pos+govt_lock2_neg+govt_lock2_neu),(govt_lock3_neu)/(govt_lock1_pos+govt_lock1_neg+govt_lock1_neu),(govt_lock4_neu)/(govt_lock4_pos+govt_lock4_neg+govt_lock4_neu)],textposition='auto')
])
# Change the bar mode
figgovt.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Reaction of the people towards Government",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1]


    )
figgovt.update_yaxes(showticklabels=True)
#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------Extend------------------------------------------------------------
d1=pd.read_csv('Hashtag Analysis/Extend Spirit/Lock1.csv')
d2=pd.read_csv('Hashtag Analysis/Extend Spirit/Lock2.csv')
d3=pd.read_csv('Hashtag Analysis/Extend Spirit/Lock3.csv')
d4=pd.read_csv('Hashtag Analysis/Extend Spirit/Lock4.csv')

ext_lock1_pos=(d1['labels'][d1['labels']==1]).count()
ext_lock1_neg=(d1['labels'][d1['labels']==-1]).count()
ext_lock1_neu=(d1['labels'][d1['labels']==0]).count()
ext_lock2_pos=(d2['labels'][d2['labels']==1]).count()
ext_lock2_neg=(d2['labels'][d2['labels']==-1]).count()
ext_lock2_neu=(d2['labels'][d2['labels']==0]).count()
ext_lock3_pos=(d3['labels'][d3['labels']==1]).count()
ext_lock3_neg=(d3['labels'][d3['labels']==-1]).count()
ext_lock3_neu=(d3['labels'][d3['labels']==0]).count()
ext_lock4_pos=(d4['labels'][d4['labels']==1]).count()
ext_lock4_neg=(d4['labels'][d4['labels']==-1]).count()
ext_lock4_neu=(d4['labels'][d4['labels']==0]).count()

lockdown=['Lockdown 1.0', 'Lockdown 2.0', 'Lockdown 3.0','Lockdown 4.0']

figext = go.Figure(data=[
    go.Bar(name='Negative', x=lockdown,y=[(ext_lock1_neg)/(ext_lock1_pos+ext_lock1_neg+ext_lock1_neu),(ext_lock2_neg)/(ext_lock2_pos+ext_lock2_neg+ext_lock2_neu),(ext_lock3_neg)/(ext_lock3_pos+ext_lock3_neg+ext_lock3_neu),(ext_lock4_neg)/(ext_lock4_pos+ext_lock4_neg+ext_lock4_neu)],textposition='auto'),
    go.Bar(name='Positive', x=lockdown,y=[(ext_lock1_pos)/(ext_lock1_pos+ext_lock1_neg+ext_lock1_neu),(ext_lock2_pos)/(ext_lock2_pos+ext_lock2_neg+ext_lock2_neu),(ext_lock3_pos)/(ext_lock3_pos+ext_lock3_neg+ext_lock3_neu),(ext_lock4_pos)/(ext_lock4_pos+ext_lock4_neg+ext_lock4_neu)],textposition='auto'),
    go.Bar(name='Neutral', x=lockdown,y=[(ext_lock1_neu)/(ext_lock1_pos+ext_lock1_neg+ext_lock1_neu),(ext_lock2_neu)/(ext_lock2_pos+ext_lock2_neg+ext_lock2_neu),(ext_lock3_neu)/(ext_lock1_pos+ext_lock1_neg+ext_lock1_neu),(ext_lock4_neu)/(ext_lock4_pos+ext_lock4_neg+ext_lock4_neu)],textposition='auto')
])
# Change the bar mode
figext.update_layout(barmode='group',
                title ="Line Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                #uniformtext_minsize=8, 
                #uniformtext_mode='hide',
                title_text="Lockdown Extension?",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                yaxis_tickformat=',.0%',
                yaxis_range=[0,1]
                

    )
figext.update_yaxes(showticklabels=True)
#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------OVERALL Sentiment-----------------------------------------------
figoverall = make_subplots(rows=1, cols=3)

negloc1 = ext_lock1_neg + govt_lock1_neg + mh_lock1_neg + eco_lock1_neg + nega_lock1_neg + ty_lock1_neg + wfh_lock1_neg
negloc2 = ext_lock2_neg + govt_lock2_neg + mh_lock2_neg + eco_lock2_neg + nega_lock2_neg + ty_lock2_neg + wfh_lock2_neg
negloc3 = ext_lock3_neg + govt_lock3_neg + mh_lock3_neg + eco_lock3_neg + nega_lock3_neg + ty_lock3_neg + wfh_lock3_neg
negloc4 = ext_lock4_neg + govt_lock4_neg + mh_lock4_neg + eco_lock4_neg + nega_lock4_neg + ty_lock4_neg + wfh_lock4_neg
posloc1 = ext_lock1_pos + govt_lock1_pos + mh_lock1_pos + eco_lock1_pos + nega_lock1_pos + ty_lock1_pos + wfh_lock1_pos
posloc2 = ext_lock2_pos + govt_lock2_pos + mh_lock2_pos + eco_lock2_pos + nega_lock2_pos + ty_lock2_pos + wfh_lock2_pos
posloc3 = ext_lock3_pos + govt_lock3_pos + mh_lock3_pos + eco_lock3_pos + nega_lock3_pos + ty_lock3_pos + wfh_lock3_pos
posloc4 = ext_lock4_pos + govt_lock4_pos + mh_lock4_pos + eco_lock4_pos + nega_lock4_pos + ty_lock4_pos + wfh_lock4_pos
neuloc1 = ext_lock1_neu + govt_lock1_neu + mh_lock1_neu + eco_lock1_neu + nega_lock1_neu + ty_lock1_neu + wfh_lock1_neu
neuloc2 = ext_lock2_neu + govt_lock2_neu + mh_lock2_neu + eco_lock2_neu + nega_lock2_neu + ty_lock2_neu + wfh_lock2_neu
neuloc3 = ext_lock3_neu + govt_lock3_neu + mh_lock3_neu + eco_lock3_neu + nega_lock3_neu + ty_lock3_neu + wfh_lock3_neu
neuloc4 = ext_lock4_neu + govt_lock4_neu + mh_lock4_neu + eco_lock4_neu + nega_lock4_neu + ty_lock4_neu + wfh_lock4_neu

#if percentage is needed just do negloc1/negloc1+negloc2+negloc3+negloc4 and so on
figoverall.add_trace(
    go.Bar(y=[negloc1,negloc2,negloc3,negloc4],
                x=lockdown,
                name='# Negative',
                textfont_color='white',
                
                #orientation='h'
                ),
    row=1, col=1
)

figoverall.add_trace(
    go.Bar(y=[posloc1,posloc2,posloc3,posloc4],
                x=lockdown,
                name='# Positive',
                textfont_color='white',
                
                #orientation='h'
                ),
    row=1, col=2
)

figoverall.add_trace(
    go.Bar(y=[neuloc1,neuloc2,neuloc3,neuloc4],
                x=lockdown,
                name='# Neutral',
                textfont_color='white',

                #orientation='h'
                ),
    row=1, col=3
)



figoverall.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Overall Analysis of the Sentiments"

                )
#----------------------------------------------------------------------------------------------
@app.callback( Output('tabs-content', 'children'),
            [Input('tabs', 'value'),
              Input('date-dropdown', 'value'),
              Input('interval-component-slow', 'n_intervals')])
def render_content(tab,sel_option,n):
    sel = dates[dates['day']==sel_option]
    global positive
    global negative
    global neutral
    # content='hi'
    # print(cal)
    try:
        df=pd.read_json('count.json')
        # print(df['details'][0])
        # print(df['dt'])
    except:
        print("cant")
    global pos
    global neg
    global neu
    global time
    global tf2
    positive=df['pos'][0]
    f ='%H:%M:%S'
    negative = df['neg'][0]
    neutral = df['neu'][0]
    cal.loc[cal['val']=='positive','count']=positive
    cal.loc[cal['val']=='negative','count']=negative
    cal.loc[cal['val']=='neutral','count']=neutral
    time=df["time"][0]
    tp=df['details'][0]
    temp={'pos':positive,'neg':negative,'neu':neutral,'time':df['time'][0],'text':[tp]}
    temp=pd.DataFrame(temp,columns=['pos','neg','neu','time','text'],index=[[1]])
    tf2=pd.concat([tf2,temp],ignore_index=True)
    
    
    figlive=px.bar(cal,x='val',y='count')
    figlive.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text="Overall Analysis of the Sentiments"

                )
    figlive2=px.scatter(tf2,x='time',y=['neg','pos','neu'])
    figlive2.update_layout( title ="Bar Chart",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a",
                plot_bgcolor="#07031a",
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                title_text=" Analysis of the Sentiments"

                )
    
    
    if tab == 'tab-1':
       return html.Div([

            html.Div([
           dcc.Graph(id='trend2',
            figure=figlive),
            
    ], style={'display':'block','padding':'0 0 0 20'}),
            html.Div([
           dcc.Graph(id='trend2',
            figure=figlive2),
], style={'display':'block','padding':'0 0 0 20'})

            ],style ={'background':'#25274d'})    
    
    elif tab == 'tab-6':
       return html.Div([
      html.Div([
                  dcc.Graph(
                      id="scatter_chart",
                      figure = {
                          'data': [
                              go.Scatter(
                                  x = sel['time'],
                                  y = sel['labels'],
                                  mode='markers'
                              )
                          ],
                          'layout' : go.Layout(
                              title = "Time Series of Sentiments throughout the Lockdown",
                              xaxis =  {'title': 'Time'},
                              yaxis = {'title': 'Sentiments'},
                              paper_bgcolor ="#07031a",
                              font=dict(
                             family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                          plot_bgcolor="#20031b",
                
                          )
                      }
                  )
        ],style={'display':'block','padding':'0 0 0 20'}),
       
       html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figty,
            
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

        html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fignega,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

       html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figwfh,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

        html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figmh,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

        
       html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figeco,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

        html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figgovt,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

       html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figext,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),
    
       html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figoverall,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),
    



            


        ],style ={'background':'#25274d'})
    elif tab == 'tab-2':
       return html.Div([
        
html.Div([
            dcc.Graph(
            id="pie_chart",
            figure=
         {
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                hole=.8,

                values=[count1,countneg,count0],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='label+percent',
                insidetextorientation='radial',
                textfont_color='white',
                marker=dict(colors=colors))],
            'layout':go.Layout(
                title ="Emotion Distribution",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a"
                )})]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),


html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig1,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig2,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),


html.Div([
                
    html.Div([html.H1('Wordcloud')],style={'color':'white','textAlign':'center','fontFamily':'Courier New,monospace'}),
    html.Div([
        html.Img(src=app.get_asset_url('lock1neg.png')),],style={'width':'50%','display':'inline-block','margin':'0 20 20 0','border':'1 px solid blue'}),
    html.Div([
        html.Img(src=app.get_asset_url('lock1pos.png')) ],style={'width':'50%','display':'inline-block','margin':'0 20 20 0','border':'1 px solid red'}),

         ]

        ,style={'margin':'0 20 20 0','background':'#07031a'}),





        ],style={'background':'#25274d'})

        
    elif tab == 'tab-3':
       return html.Div([
        
html.Div([
            dcc.Graph(
            id="pie_chart",
            figure=
         {
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                hole=.8,

                values=[count1loc2,countneg2,count0loc2],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='label+percent',
                insidetextorientation='radial',
                textfont_color='white',
                marker=dict(colors=colors))],
            'layout':go.Layout(
                title ="Emotion Distribution",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a"
                )})]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),


html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig1loc2,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figloc2,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig2loc2,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),


html.Div([ 
     html.Div([html.H1('Wordcloud')],style={'color':'white','textAlign':'center','fontFamily':'Courier New,monospace'}),
      html.Div([
        html.Img(src=app.get_asset_url('lock2neg.png')),],style={'width':'50%','display':'inline-block','margin':'0 20 20 0','border':'1 px solid blue'}),
    html.Div([
        html.Img(src=app.get_asset_url('lock2pos.png')) ],style={'width':'50%','display':'inline-block','margin':'0 20 20 0','border':'1 px solid red'}),

         ]   

         ,style={'margin':'0 20 20 0','background':'#07031a'}),





        ],style={'background':'#25274d'})

        
    elif tab == 'tab-4':
       return html.Div([
        
html.Div([
            dcc.Graph(
            id="pie_chart",
            figure=
         {
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                hole=.8,

                values=[count1loc3,countneg3,count0loc3],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='label+percent',
                insidetextorientation='radial',
                textfont_color='white',
                marker=dict(colors=colors))],
            'layout':go.Layout(
                title ="Emotion Distribution",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a"
                )})]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),


html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig1loc3,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figloc3,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig2loc3,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),


html.Div([
     html.Div([html.H1('Wordcloud')],style={'color':'white','textAlign':'center','fontFamily':'Courier New,monospace'}),
   html.Div([
        html.Img(src=app.get_asset_url('lock3neg.png')),],style={'width':'50%','display':'inline-block','margin':'0 20 20 0','border':'1 px solid blue'}),
    html.Div([
        html.Img(src=app.get_asset_url('lock3pos.png')) ],style={'width':'50%','display':'inline-block','margin':'0 20 20 0','border':'1 px solid red'}),

         ]

        ,style={'margin':'0 20 20 0','background':'#07031a'}),





        ],style={'background':'#25274d'})

        
    elif tab == 'tab-5':
       return html.Div([
        
html.Div([
            dcc.Graph(
            id="pie_chart",
            figure=
         {
            'data':[
            go.Pie(
                labels=['positives','negatives','neutrals'],
                hole=.8,

                values=[count1loc4,countneg4,count0loc4],
                name="Sentiment Analysis",
                hoverinfo='label+percent',
                textinfo='label+percent',
                insidetextorientation='radial',
                textfont_color='white',
                marker=dict(colors=colors))],
            'layout':go.Layout(
                title ="Emotion Distribution",
                font=dict(
                    family="Courier New,monospace",
                    size=14,
                    color='white'

                      ),

                paper_bgcolor ="#07031a"
                )})]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),


html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig1loc4,
         )]
            ,style={'width':'50%','display':'inline-block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=figloc4,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),

html.Div([
            dcc.Graph(
            id="bar_chart",
            figure=fig2loc4,
         )]
            ,style={'display':'block','padding':'0 0 0 20'}),


html.Div([ 
        html.Div([html.H1('Wordcloud')],style={'color':'white','textAlign':'center','fontFamily':'Courier New,monospace'}),
    html.Div([
        html.Img(src=app.get_asset_url('lock4neg.png'))],style={'width':'50%','display':'inline-block','padding':'0 0 0 0'}),
        html.Div([
        html.Img(src=app.get_asset_url('lock4pos.png')) ],style={'width':'50%','display':'inline-block','padding':'0 0 0 0'})

         ]

        ,style={'background':'#07031a'}
        ),





        ],style={'background':'#25274d'})

if __name__ == '__main__':
    app.run_server( port=port,debug=True)
