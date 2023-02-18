import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_datareader.yahoo.daily import YahooDailyReader
from datetime import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from neuralprophet import NeuralProphet         #NeuralProphet
from sklearn.metrics import mean_absolute_error #評価指標MAE
from statistics import mean                     #平均値の計算
import plotly.offline as py

#########################################################
# 学習データ整備
#########################################################
csv_path = './data/btc.csv'
df = pd.read_csv(csv_path)
df['date(Y/M/D H)'] = pd.to_datetime(df['date(Y/M/D H)'], format='%Y/%m/%d %H')

# 元の桁が大きすぎるため、10000で割ることで数値を最小限にする
col_names = ['last(JPY)','high(JPY)','low(JPY)']
# ave = (df['last(JPY)'].max() + df['last(JPY)'].min())/2
for col in col_names:
	# df[col] -= ave
	df[col] /= 10000

# 列名の変更
data = df.reset_index().rename(columns={'date(Y/M/D H)': 'ds', 'last(JPY)': 'y'})

# 訓練データ・検証データ
print("data: " + str(len(data)))
train_len = int(len(data) * 0.7)

df_train = data[:train_len]
df_test = data[train_len:]
train_len = len(df_train)
test_len = len(df_test)
print("df_train: " + str(train_len)+", df_test: " + str(test_len))

#########################################################
# 学習
#########################################################
# インスタンス化
model = Prophet()
# 学習
model.fit(df_train)


#########################################################
# 予測
#########################################################
# 学習データに基づいて未来を予測(検証h分)
future = model.make_future_dataframe(periods=test_len,freq='H')
forecast = model.predict(future)

#########################################################
# 精度評価
#########################################################
# テストデータに予測値を結合
# df_test['Prophet Predict'] = forecast[-train_len:]['yhat']
df_test = df_test.merge(forecast[['ds', 'yhat']], on='ds')
print(df_test)

#########################################################
# 結果表示
#########################################################
print('MAE:')
print(mean_absolute_error(df_test['y'], df_test['yhat'])) 
print('MAPE:')
print(mean(abs(df_test['y'] - df_test['yhat'])/df_test['y']) *100)

## 予測結果の可視化
## 描画
#fig1 = plot_plotly(model, forecast)
## ノードブック上に出力
#py.iplot(fig1)




