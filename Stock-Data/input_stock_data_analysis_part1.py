#https://medium.com/@alexzap922/backtesting-algo-trading-strategies-fintech-analysis-portfolio-optimization-nvda-amd-intc-25c5ab5c768a
import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd() 
#!pip install yfinance, plotly, scipy, matplotlib, quantstats, ta, datetime, seaborn, pandas_ta 
#!pip install pyfolio-reloaded 
#Basic Imports
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import quantstats as qs
import ta
from datetime import datetime as dt, timedelta as td
import seaborn as sns
import pandas_ta as taa
import pyfolio as pf
#Reading Historical Stock Data
tickers_list = ['NVDA', 'INTC', 'AMD', 'MSI']
start_date = '2023-01-01'

data0 = yf.download(tickers_list, start=start_date)
data0.tail()

data0.shape
data0.info()
data0. isnull().values.any()
data0.describe().T

tickers_list = ['NVDA', 'INTC', 'AMD', 'MSI']
data = yf.download(tickers_list,'2023-1-1')['Adj Close']
print(data.tail())

# Plotting candlestick chart without indicators
from plotly.subplots import make_subplots
import plotly.graph_objects as go
start='2023-01-01'
# Downloading Apple Stocks
nvda = yf.download('NVDA', start = start)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights = [0.7, 0.3])
fig.add_trace(go.Candlestick(x=nvda.index,
                             open=nvda['Open'],
                             high=nvda['High'],
                             low=nvda['Low'],
                             close=nvda['Adj Close'],
                             name='NVDA'),
              row=1, col=1)


# Plotting volume chart on the second row 
fig.add_trace(go.Bar(x=nvda.index,
                     y=nvda['Volume'],
                     name='Volume',
                     marker=dict(color='orange', opacity=1.0)),
              row=2, col=1)

# Plotting annotation
fig.add_annotation(text='NVDA',
                    font=dict(color='white', size=40),
                    xref='paper', yref='paper',
                    x=0.5, y=0.65,
                    showarrow=False,
                    opacity=0.2)

# Configuring layout
fig.update_layout(title='NVDA Candlestick Chart',
                  yaxis=dict(title='Price (USD)'),
                  height=1000,
                 template = 'plotly_dark')

# Configuring axes and subplots
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
fig.update_yaxes(title_text='Volume', row=2, col=1)

fig.show()

nvda = yf.Ticker("NVDA").history(period='2y')['Close']
amd= yf.Ticker("AMD").history(period='2y')['Close']
msi= yf.Ticker("MSI").history(period='2y')['Close']
intc=yf.Ticker("INTC").history(period='2y')['Close']

#NVDA

print(nvda.mean())
print(nvda.median())
print(nvda.min())
print(nvda.max())
print(nvda.std())
print(nvda.mode())
nvda.describe()

#AMD

print(amd.mean())
print(amd.median())
print(amd.min())
print(amd.max())
print(amd.std())
print(amd.mode())
amd.describe()

#MSI

print(msi.mean())
print(msi.median())
print(msi.min())
print(msi.max())
print(msi.std())
print(msi.mode())
msi.describe()

#INTC

print(intc.mean())
print(intc.median())
print(intc.min())
print(intc.max())
print(intc.std())
print(intc.mode())
intc.describe()

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Read Data
# Define the ticker list
import pandas as pd
tickers_list = ['NVDA', 'INTC', 'AMD', 'MSI']

# Fetch the data
import yfinance as yf
data = yf.download(tickers_list,'2023-1-1')['Adj Close']

# Print first 5 rows of the data
print(data.tail())

# Volatility of 4 stocks
data.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250)).plot(kind='bar')

# Log of percentage change cov
cov_matrix = data.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix

corr_matrix = data.pct_change().apply(lambda x: np.log(1+x)).corr()
corr_matrix

# Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
ann_sd = data.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
ann_sd

#Correlations

data['NVDA'].corr(data['AMD'])

data['NVDA'].corr(data['MSI'])

data['NVDA'].corr(data['INTC'])


# Yearly returns for individual companies
ind_er = data.resample('YE').last().pct_change().mean()
ind_er

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import pandas_datareader as web
from datetime import datetime as dt, timedelta as td
end = dt.today()
start = end - td(days=2*365)

stocks = ['NVDA', 'INTC', 'AMD', 'MSI','^GSPC']

# Fetch the data
import yfinance as yf
df = yf.download(stocks,start=start,end=end)['Adj Close']

px.line(df * 100 / df.iloc[0])

ret_port = df.pct_change()
px.line(ret_port)

cumretport = (1 + ret_port).cumprod() - 1 
px.line(cumretport*100)

df_return = np.log(df / df.shift())
(df_return.mean() * 12).plot.bar()
plt.title(f'Annual log return')
cov = df_return.cov() * 12
market_cov = cov.iloc[0,1]
var_market = cov.iloc[1,1]
beta = market_cov / var_market
print (beta)
