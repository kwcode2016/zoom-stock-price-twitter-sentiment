import pandas_datareader.data as web
import pandas as pd
import datetime as dt

start_date = '2019-04-01'
end_date = '2023-04-16'
output_filename = 'zoom-stock-prices-' + start_date + '-to-' + end_date + '.csv'

print(output_filename)
df = web.DataReader('ZM', 'yahoo', start=start_date, end=end_date)
df.to_csv(output_filename)
