import streamlit as st
import pickle
import numpy as np
#import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Laptop Predictor')

#brand
company = st.selectbox('Brand',df['Company'].unique())

#Type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

#Ram
ram =st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

#wEIGT
weight =st.number_input('Weight of the laptop')

#Touchscreen 
touchscreen = st.selectbox('Touch Screen',['NO','YES'])

#IPS
ips= st.selectbox('IPS',['NO','YES'])

#sCREEN RESO
screen_size = st.number_input('Screen Size')

#reso
reso = st.selectbox('Screen resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
                                         '2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU Brand',df['cpu brand'].unique())

#hdd
hdd = st.selectbox('HDD(in GB)',[0,128,258,512,1024,2048])

#ssd
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#GPU
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

#OS
os = st.selectbox('OS',df['os'].unique())

if st.button('predict Price'):
    ppi=None
    if touchscreen=='YES':
        touchscreen=1
    else:
        touchscreen=0
    
    if ips=='YES':
        ips=1
    else:ips=0

    x_res = int(reso.split('x')[0])
    y_res = int(reso.split('x')[1])

    ppi = (x_res**2 + y_res**2)**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    output = np.exp(pipe.predict(query)[0])
    st.title('The predicted price for this configuration is Rs. '+'%.2f' % output)