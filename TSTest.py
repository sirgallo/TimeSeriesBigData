from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
#from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import *
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import pandas as pd
import pmdarima as pm

##  initialize spark environment
try:
    sc.stop()
except:
    pass

#local[*] or ("spark://192.168.1.109:7077")
conf = SparkConf().setAppName('TimeSeriesProjectTest').setMaster('local[*]')
sc = SparkContext(conf = conf)
spark = SparkSession(sparkContext = sc)
#sc.setLogLevel("ERROR")
sc.setLogLevel('OFF')

########
########

##  functions

def decompose(data, dates, period, model = None):
  ##  decompose the data to reveal trend, seasonality, and residuals (white noise)
  decomposition = 0
  if model != None:
    decomposition = seasonal_decompose(data, model = model, period = period)
  else:
    decomposition = seasonal_decompose(data, period = period)
  trendData = decomposition.trend
  seasonalData = decomposition.seasonal
  observedData = decomposition.observed
  residData = decomposition.resid

  ##  plot the decomposed data to look at trend, seasonality, and residuals
  fig, axs = plt.subplots(4)
  fig.suptitle('Trend Data, Seasonal Data, Observed Data, and Residuals')
  axs[0].plot(dates, trendData)
  axs[1].plot(dates, seasonalData)
  axs[2].plot(dates, observedData)
  axs[3].plot(dates, residData)
  plt.show()

def acfPacf(data, type, lags):
  ##  find the acf and the pacf
  datapacf = pacf(data)
  dataacf = acf(data)

  ##  need to prepare data to be charted using acf/pacf charting functions
  chartdata = spark.createDataFrame(data, type).toPandas()
  
  ##  plot the acf and pacf with the specified number of data lags
  fig, ax = plt.subplots(2, 1)
  fig = plot_acf(chartdata, lags = lags, ax = ax[0])
  fig = plot_pacf(chartdata, lags = lags, ax = ax[1])
  plt.show()

def checkStationarity(data, maxar, maxma, ic, trend):
  ##  use built in ARMA function to guess the p and q values in ARMA(p, q)
  bestfit = sm.tsa.arma_order_select_ic(data, max_ar = maxar, max_ma = maxma, ic = ic, trend = trend)
  print('best fit (p, q) for data: ', bestfit['aic_min_order'])

  return bestfit['aic_min_order']

def dataDiff(data, period = 1):
  ##  difference the data with regard to seasonality
  ##  period works by selecting a set of days over which the data cycles
  diff = []
  for i in range(period, len(data)):
    val = abs(data[i] - data[i - period])
    diff.append(val)

  plt.plot(diff)
  plt.show()

  return diff

def invertDiff(diff, data, period = 1):
  inv = []
  for i in range(len(diff)):
    val = data[i - period] + diff[i]
    inv.append(val)
  
  plt.plot(inv)
  plt.show()

  return inv

def invertData(yhat, data, period = 1):
  return yhat + data[-period]

def meanSquareError(actual, prediction):
  sum = 0
  for i in range(len(actual)):
    sum = sum + ((actual[i] - prediction[i]) ** 2)
  
  mse = sum/len(actual)
  
  return mse

def buildModelARIMA(train, test, order, seasorder, period, lags, exog, dynamic):
  ##  fit model with specific order
  model = ARIMA(train, order = order)
  modelfit = model.fit()
  print(modelfit.summary())

  resid = modelfit.resid
  fig, ax = plt.subplots(2, 1)
  fig = plot_acf(resid, lags = lags, ax = ax[0])
  fig = plot_pacf(resid, lags = lags, ax = ax[1])
  plt.show()
  
  return(modelfit)

def buildModelSARIMA(train, startp, startq, test, maxp, maxq, m):
  ##  build SARIMA model
  ##  use the train data set and find values for (p, d, q) (P, D, Q, m)
  ##  m is a known and is the seasonality that the data follows
  ##  test is the test used to determine coefficients and the order of the components
  smodel = pm.auto_arima(train, start_p = startp, start_q = startq, 
                    test = test, max_p = maxp,
                    max_q = maxq, m = m,
                    start_P = 0, seasonal = True,
                    d = 1, D = 1, trace = True,
                    error_action = 'ignore',
                    suppress_warnings = True,
                    stepwise = True)
  
  smodel.summary()

  return smodel

def predictFutureARIMA(modelfit, test, start, end, dynamic):
  ##  fit the model to the trainset and plot for visual accuracy
  modelfit.plot_predict(dynamic = False)
  plt.show()

  ##  forecast future data on the training set using the fit model, forecasting as far as the test set
  ##  alpha is the confidence interval and is set to 95%
  fc, se, conf = modelfit.forecast(len(test), alpha = 0.05)
  
  ##  plot the forecast with the confidence interval
  plt.plot(fc)
  plt.plot(conf[:, 0], color = 'green')
  plt.plot(conf[:, 1])
  plt.plot(test)
  plt.show()

  return [fc, conf[:, 0], conf[:, 1]]

def predictFutureSARIMA(model, test, periods):
  fitted, confint = model.predict(n_periods = periods, return_conf_int = True)

  plt.plot(test)
  plt.plot(fitted)
  plt.show()

########
########

##  define schema and read in both test and training sets for this particular dataset
weatherschema = StructType([StructField('date', DateType(), True),
                            StructField('meantemp', DoubleType(), True),
                            StructField('humditiy', DoubleType(), True),
                            StructField('wind_speed', DoubleType(), True),
                            StructField('meanpressure', DoubleType(), True)
                          ])
                            
##  read from project repo
testdf = spark.read.csv('./data/DailyDelhiClimateTest.csv', schema = weatherschema, header = True)
traindf = spark.read.csv('./data/DailyDelhiClimateTrain.csv', schema = weatherschema, header = True)
totaldf = spark.read.csv('./data', schema = weatherschema, header = True)

##  some useful information from the dataset
totaldf.printSchema()
totaldf.show()

########
########

## important features from total
totaldf.createOrReplaceTempView('climate')
totalmeantemplist = spark.sql('select meantemp from climate').rdd.map(lambda row: row[0]).collect()
totaldateslist = spark.sql('select date from climate').rdd.map(lambda row: row[0]).collect()

##  decompose the data to look for seasonality and trend, as well as white noise
decompose(totalmeantemplist, totaldateslist, 365, 'multiplicative')

##  important features from trainset
traindf.createOrReplaceTempView('climatetrain')
trainmeantemplist = spark.sql('select meantemp from climatetrain').rdd.map(lambda row: row[0]).collect()
traindateslist = spark.sql('select date from climatetrain').rdd.map(lambda row: row[0]).collect()
trainmeantempstruclist = spark.sql('select meantemp from climatetrain').collect()

##  important features from testset
testdf.createOrReplaceTempView('climatetest')
testmeantemplist = spark.sql('select meantemp from climatetest').rdd.map(lambda row: row[0]).collect()
testdateslist = spark.sql('select date from climatetest').rdd.map(lambda row: row[0]).collect()

#decompose(trainmeantemplist, traindateslist, 365, 'multiplicative')

##  plot acf/pacf of differenced data to check for order for ARIMA model (p, d, q)
acfPacf(trainmeantemplist, DoubleType(), lags = 380)

##  find the best fit for the data for (p, q)
bestfit = checkStationarity(trainmeantemplist, 5, 5, 'aic', 'c')

##  take the first difference
meantempdiff1st = dataDiff(trainmeantemplist, 1)
acfPacf(meantempdiff1st, DoubleType(), 380)

##  take a difference in regards to a seasonal period
meantempdiffseas = dataDiff(trainmeantemplist, 365)
acfPacf(meantempdiffseas, DoubleType(), 380)

##  take first order diff with seasonal comp
meantempdiff1stseas = dataDiff(meantempdiff1st, 365)
acfPacf(meantempdiff1stseas, DoubleType(), 380)

##  invert the data
invdiff = invertDiff(meantempdiff1st, trainmeantemplist)
print(invdiff)

##  predict future data by fitting an ARIMA model and testing against the test data
arima = buildModelARIMA(trainmeantemplist, testmeantemplist, order = (3, 1, 2), seasorder = (0, 0, 0, 0), period = 365, lags = 50, exog = None, dynamic = True)
results = predictFutureARIMA(arima, testmeantemplist, len(trainmeantemplist) - len(testmeantemplist), len(testmeantemplist), True)

##  test actual vs predicted error with mean square error
mse = meanSquareError(testmeantemplist, results[1])
print('The Mean Square Error is: ', mse)

##  predict future data by fitting an SARIMA model
#sarima = buildModel(trainmeantemplist, testmeantemplist, order = (3, 1, 2))
sarima = buildModelSARIMA(trainmeantemplist, 1, 1, 'aic', 5, 5, 12)
predictFutureSARIMA(sarima, testmeantemplist, 100)

sc.stop()