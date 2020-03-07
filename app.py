# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for, render_template_string,send_from_directory
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg') #これを入れないとエラー

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

showcols=[] 
#uploadfile=[] 
SAVE_DIR='./static'
app.config['UPLOAD_FILE'] = ""

#データはここで読み込んでおく
#df=pd.read_csv("sampledata/iris.csv") #あとでデータセットを指定したい

# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@app.route('/')
def index():
    #showcols=df.columns
    if len(os.listdir(SAVE_DIR))==0:
        images=[]   
    else:
        images=sorted(os.listdir(SAVE_DIR))[-1]

    return render_template('index.html',showcols=showcols,uploadfile=app.config['UPLOAD_FILE'],images=images)

#データをアップロードした時の処理
@app.route('/upload',methods=['POST','GET'])
def upload():
    #グローバルにしたい
    csvdata = request.files.get('csvfile')
    app.config['UPLOAD_FILE'] =csvdata
    #ファイルを保存
    global df
    df=pd.read_csv(csvdata) #列を読み込むだけ
    showcols.extend(list(df.columns))
    return redirect('/')
    #raise render_template('index.html',showcols="bbb")

@app.route('/selectcol',methods=['GET', 'POST'])
def selectcol():
    col = request.form.get('datacolumns')
    #print(col)
    #print(df.shape)
    distributions=[st.norm,st.gamma,st.rayleigh,st.uniform,]

    result=[]
    for distribution in distributions:
        y, x = np.histogram(df[col], bins=20, density=True)
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
        #print(result)
        
        #savefig
    plt.figure()
    for i in range(len(result)):
        pd.Series(result[i][0], x).plot(label=distributions[i].name)
    
    plt.hist(df[col],density=True,alpha=0.4,bins=20)
    plt.legend()
    dt_now = datetime.now().strftime("%Y%m%d%_H%M%S")
    plt.savefig(f"{SAVE_DIR}/{dt_now}.png")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0') # どこからでもアクセス可能に

"""
#データ列を選択した時の画像表示処理
@app.route('/selectcol',methods=['GET', 'POST'])
def selectcol():
    select = request.form.get('datacolumns')
    #return(str(select)) 
    return render_template('index.html',showcols=showcols)
"""
"""


@app.route('/data')
def data():
    return render_template_string('showdata.html', table=df.to_html(header='true'))



# /post にアクセスしたときの処理
@app.route('/post', methods=['GET', 'POST'])
def post():
    title = "こんにちは"
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して
        name = request.form['name']
        # index.html をレンダリングする
        return render_template('index.html',
                               name=name, title=title)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))
 


if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0') # どこからでもアクセス可能に

"""