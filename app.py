from flask import Flask, render_template, request, redirect, url_for, render_template_string,send_from_directory,abort,session
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg') #これを入れないとエラー

#確率分布の辞書作成
dist_d={}
dist_d["norm"]=st.norm
dist_d["gamma"]=st.gamma
dist_d["rayleigh"]=st.rayleigh
dist_d["beta"]=st.beta
dist_d["chi2"]=st.chi2

app = Flask(__name__)
app.secret_key = 'testtest'
SAVE_DIR='./static'

@app.route('/')
def index():
    return render_template('index.html')
    
#データをアップロードした時の処理
@app.route('/upload',methods=['POST','GET'])
def upload():
    csvdata = request.files['csvfile']
    session["UPLOAD_FILE"] = csvdata.filename
    
    global df
    df=pd.read_csv(csvdata)
    session["SELECT_COLS"] = list(df.columns)
    return render_template('index.html',showcols=session["SELECT_COLS"],uploadfile=session["UPLOAD_FILE"])
    #return redirect('/')

@app.route('/selectcol',methods=['GET', 'POST'])
def selectcol():
    col = request.form.get('datacolumns')
    usedists = request.form.getlist("dist")
    distributions=[dist_d[i] for i in usedists]

    try:
        result=[]
        for distribution in distributions:
            y, x = np.histogram(df[col], bins='auto', density=True) #binはautoでよい？
            x=np.convolve(x, np.ones(2)/2, mode='same')[1:]
            
            params = distribution.fit(df[col])

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
            
            result.append((pdf,sse))

        global result_table
        result_table=pd.DataFrame()
        result_table["name"]=usedists
        result_table["sse"]=[i[1] for i in result]
        result_table.sort_values(by="sse",inplace=True)

        plt.figure()
        for i in range(len(result)):
            pd.Series(result[i][0], x).plot(label=distributions[i].name)
        
        plt.hist(df[col],density=True,alpha=0.4,bins="auto")
        plt.legend()
        dt_now = datetime.now().strftime("%Y%m%d%_H%M%S")
        plt.savefig(f"{SAVE_DIR}/{dt_now}.png")
        session["IMAGE_FILE"]=f"{dt_now}.png"
    
    except:
        abort(404)
    
    return render_template('index.html',showcols=session["SELECT_COLS"],uploadfile=session["UPLOAD_FILE"],images=session["IMAGE_FILE"],tables=[result_table.to_html(classes='data')])

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')