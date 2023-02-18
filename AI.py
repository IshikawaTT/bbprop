import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_datareader.yahoo.daily import YahooDailyReader
from datetime import datetime
from fbprophet import Prophet
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
csv_path = './data/btc.csv'
df = pd.read_csv(csv_path)
df.drop_duplicates(subset=['date(Y/M/D H)'], inplace=True)
df = df.reset_index()
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

# 説明変数情報
regressorsList = []
for col in data.columns[4:]:
	regressorsList.append(str(col))
print(regressorsList)


#########################################################
# 学習
#########################################################
print("# 学習 ###########################################")
# インスタンス化
model = NeuralProphet(
	#growth="linear",
	#changepoints=None,
	#n_changepoints=10,
	#changepoints_range=0.9,
	#trend_reg=0,
	#trend_reg_threshold=False,
	#yearly_seasonality="auto",
	#weekly_seasonality="auto",
	#daily_seasonality="auto",
	seasonality_mode="multiplicative",
	#seasonality_reg=0,
	#n_forecasts=1,
	n_lags=1,
	num_hidden_layers=1,
	d_hidden=1,
	#ar_reg=None,
	#learning_rate=None,
	#epochs=None,
	#batch_size=None,
	#loss_func="Huber",
	#optimizer="AdamW",
	#newer_samples_weight=2,
	#newer_samples_start=0.0,
	#impute_missing=True,
	#collect_metrics=True,
	#normalize="auto",
	#global_normalization=False,
	#global_time_normalization=True,
	#unknown_data_normalization=False,
)
model.add_lagged_regressor(names=regressorsList)


# 学習
#model.fit(df_train[['ds', 'y']], freq="H")
df_train_tmp = df_train.drop(['level_0','index'],axis='columns')
model.fit(df_train_tmp, freq="H")

#########################################################
# 予測
#########################################################
print("# 予測 ###########################################")
# 学習データに基づいて未来を予測(検証h分)
future = model.make_future_dataframe(df_train_tmp, periods=test_len, n_historic_predictions=train_len)
forecast = model.predict(future)
print(forecast)

#########################################################
# 精度評価
#########################################################
print("# 精度評価 #######################################")
# テストデータに予測値を結合
# df_test['Prophet Predict'] = forecast[-train_len:]['yhat']
df_test = df_test.merge(forecast[['ds', 'yhat1']], on='ds')

#########################################################
# 結果表示
#########################################################
print("# 結果表示 #######################################")
print('RMSE:'+ str(np.sqrt(mean_squared_error(df_test['y'], df_test['yhat1']))) )
print('MAE:'+ str(mean_absolute_error(df_test['y'], df_test['yhat1'])) )
print('MAPE:'+ str(mean(abs(df_test['y'] - df_test['yhat1'])/df_test['y']) *100) )



