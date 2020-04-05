import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
import streamlit as st

#確率分布の辞書作成
dist_d={}
dist_d["norm"]=stat.norm
dist_d["lognorm"]=stat.lognorm
dist_d["gamma"]=stat.gamma
#dist_d["rayleigh"]=stat.rayleigh
dist_d["beta"]=stat.beta
dist_d["chi2"]=stat.chi2
dist_d["f"]=stat.f
dist_d["t"]=stat.t
dist_d["cauchy"]=stat.cauchy
dist_d["uniform"]=stat.uniform
distlist=list(dist_d.keys())

st.title('Distribution Fitter')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if st.checkbox('Show dataframe'):
        st.write(data)

    columns=tuple(data.columns)
    col = st.sidebar.radio(
        "select column",
        columns)

    usedists = st.sidebar.multiselect(
        'select probability distribution',
        distlist)

    distributions=[dist_d[i] for i in usedists]

    if st.checkbox('select the number of bins manually'):
        binnumber = st.slider('set bin', 0, data.shape[0]//5 , data.shape[0]//10)
    else:
        binnumber = "auto"

    result=[]
    for distribution in distributions:
        y, x = np.histogram(data[col], bins=binnumber, density=True) #binはautoでよい？
        x=np.convolve(x, np.ones(2)/2, mode='same')[1:]
        
        params = distribution.fit(data[col])

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))
        
        result.append((pdf,sse))
       
    result_table=pd.DataFrame()
    result_table["name"]=usedists
    result_table["sse"]=[i[1] for i in result]
    result_table.sort_values(by="sse",inplace=True)
    
    plt.figure()
    for i in range(len(result)):
        pd.Series(result[i][0], x).plot(label=distributions[i].name)
    
    plt.hist(data[col],density=True,alpha=0.4,bins=binnumber)
    plt.legend()
    st.pyplot()
    st.write(result_table)
   