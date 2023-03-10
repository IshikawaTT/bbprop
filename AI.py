import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_datareader.yahoo.daily import YahooDailyReader
from datetime import datetime
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statistics import mean
import plotly.offline as py

#########################################################
# 学習データ整備
#########################################################
print("# 学習データ整備##################################")
coin_name='btc'
csv_path = './data/' + coin_name + '.csv'
df = pd.read_csv(csv_path)
df = df[200:]
df.drop_duplicates(subset=['date(Y/M/D H)'], inplace=True)
df = df.reset_index()
df['date(Y/M/D H)'] = pd.to_datetime(df['date(Y/M/D H)'], format='%Y/%m/%d %H')

if coin_name == 'btc':
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
train_len = int(len(data) * 0.9)

df_train = data[:train_len]
df_test = data[train_len:]
train_len = len(df_train)
test_len = len(df_test)
print("df_train: " + str(train_len)+", df_test: " + str(test_len))

# 説明変数情報
regressorsList = []
for col in data.columns[4:]:
	regressorsList.append(str(col))
print("対象: " + coin_name)
print("説明変数情報:" + str(regressorsList))


#########################################################
# 学習
#########################################################
print("# 学習 ###########################################")
# インスタンス化
model = NeuralProphet(
	growth="off",
	#changepoints=None,
	#n_changepoints=10,
	#changepoints_range=0.9,
	#trend_reg=0,
	#trend_reg_threshold=False,
	yearly_seasonality=False,
	#weekly_seasonality="auto",
	#daily_seasonality="auto",
	seasonality_mode="multiplicative",
	#seasonality_reg=0,
	#n_forecasts=test_len,
	#n_lags=0,
	#num_hidden_layers=None,
	#d_hidden=None,
	#ar_reg=None,
	#learning_rate=None,
	#epochs=None,
	#batch_size=None,
	#loss_func="Huber",
	#optimizer="AdamW",
	newer_samples_weight=5,
	#newer_samples_start=0.0,
	#impute_missing=True,
	#collect_metrics=True,
	#normalize="auto",
	#global_normalization=False,
	#global_time_normalization=True,
	#unknown_data_normalization=False,
	normalize="standardize",
)
for reg_name in regressorsList:
	model.add_future_regressor(
		name=reg_name,
		normalize="true",
	)


# 学習
#model.fit(df_train[['ds', 'y']], freq="H")
df_train_tmp = df_train[['ds', 'y']+regressorsList]
model.fit(
	df_train_tmp,
	freq="H",
	progress="bar",
)

#########################################################
# 予測
#########################################################
print("# 予測 ###########################################")
# 学習データに基づいて未来を予測(検証h分)
future = model.make_future_dataframe(
	df_train_tmp,
	#events_df=None,
	regressors_df=df_train[regressorsList] if len(regressorsList) > 0 else None,
	periods=test_len,
	n_historic_predictions=train_len,
)
forecast = model.predict(future)
print(forecast)
#########################################################
# 精度評価
#########################################################
print("# 精度評価 #######################################")
# テストデータに予測値を結合
# df_test['Prophet Predict'] = forecast[-train_len:]['yhat']
df_test = df_test.merge(forecast[['ds', 'yhat1']], on='ds')
print(df_test)
print('RMSE:'+ str(np.sqrt(mean_squared_error(df_test['y'], df_test['yhat1']))) )
print('MAE:'+ str(mean_absolute_error(df_test['y'], df_test['yhat1'])) )
print('MAPE:'+ str(mean(abs(df_test['y'] - df_test['yhat1'])/df_test['y']) *100) )

print("# グラフ生成 #####################################")
fig = model.plot(df_test)
plt.show()


