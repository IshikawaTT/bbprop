import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_datareader.yahoo.daily import YahooDailyReader
from datetime import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py

#########################################################
# 学習データ整備
#########################################################
csv_path = 'C:/Users/Tsubasa/work/bitbank/data/btc.csv'
df = pd.read_csv(csv_path)
df['date(Y/M/D H)'] = pd.to_datetime(df['date(Y/M/D H)'], format='%Y/%m/%d %H')

# 元の桁が大きすぎるため、10000で割ることで数値を最小限にする
col_names = ['last(JPY)','high(JPY)','low(JPY)']
# ave = (df['last(JPY)'].max() + df['last(JPY)'].min())/2
for col in col_names:
	# df[col] -= ave
	df[col] /= 10000

# 学習データの先頭5つ確認
print(df.head()) 

# 列名の変更
data = df.reset_index().rename(columns={'date(Y/M/D H)': 'ds', 'last(JPY)': 'y'})
print(data.head()) 
# インスタンス化
model = Prophet()
model.add_regressor('low(JPY)')
# 学習
model.fit(data)

# 学習データに基づいて未来を予測(24h分)
future = model.make_future_dataframe(periods=24,freq='H')
forecast = model.predict(future)

print(forecast.tail(30))

# 予測結果の可視化
# 描画
#fig1 = plot_plotly(model, forecast)
# ノードブック上に出力
#py.iplot(fig1)


