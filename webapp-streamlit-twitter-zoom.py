import streamlit as st
import pandas as pd


# creating the side bar
# status_text = st.sidebar.empty()

# st.sidebar.markdown("""# hello sidebar
# some things will go here

#                     """)


st.set_page_config(page_title='Zoom Stock Twitter Analysis')



st.sidebar.write('turn on each chart check marks')
st.sidebar.write('[ ] Zoom Stock Price')
st.sidebar.write('[ ] Positive sentiment [blue box]')
st.sidebar.write('[ ] Neutral Sentiment Tweets [grey box]')
st.sidebar.write('[ ] Negative Sentiment Tweets [red box]')
st.sidebar.write('date slider')

start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")


st.title("Zoom Stock Prices with Twitter Sentiment Analysis")


st.write("Stock Price Chart")
zm_stock_df = pd.read_csv('zoom-stock-prices-2019-04-01-2022-12-11.csv')

# zm_stock_df[400:450]

# zm_stock_df[-100:]

zm_date_close = zm_stock_df[['Date', 'Close']]


zm_date_close['Date'] = pd.to_datetime(zm_date_close['Date'])
print(zm_date_close.info())

# st.write(zm_date_close)


st.line_chart(zm_date_close, x='Date', y='Close')



st.title('testing dual y axis with altair')

import altair as alt

df = pd.DataFrame({
    'name': ['brian', 'dominik', 'patricia'],
    'age': [20, 30, 40],
    'salary': [100, 50, 300]
})

a = alt.Chart(df).mark_area(opacity=1).encode(
    x='name', y='age')

b = alt.Chart(df).mark_area(opacity=0.6).encode(
    x='name', y='salary')

c = alt.layer(a, b)

st.altair_chart(c, use_container_width=True)
#









import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

chart_data = pd.DataFrame(
    np.random.randn(30, 5),
    columns=['a', 'b', 'c', 'd', 'e'])

c = alt.Chart(chart_data).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

st.altair_chart(c, use_container_width=True)


from vega_datasets import data

source = data.seattle_weather()

source

# base = alt.Chart(source).encode(alt.X('month(date):T').axis(title=None))
base = alt.Chart(source).encode(
    alt.X('month(date):T')
)

area = base.mark_area(opacity=0.3, color='#57A44C').encode(
    alt.Y('average(temp_max)', title='test1 temp'),
    alt.Y2('average(temp_min)', title='test2 temp2')
)

line = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
    alt.Y('average(precipitation)', title='Precipitation (inches)', titleColor='#5276A7')
)

alt.layer(area, line).resolve_scale(
    y='independent'
)


# st.altair_chart(source['date','precipitation','temp_max', 'temp_min'])


st.write ("END")


import altair as alt
from vega_datasets import data

iris = data.iris()




chart1 = alt.Chart(iris).mark_point().encode(
    x='petalLength:Q',
    y='petalWidth:Q',
    color='species:N'
).properties(
    height=300,
    width=300
).interactive()

chart2 = alt.Chart(iris).mark_bar().encode(
    x='count()',
    y=alt.Y('petalWidth:Q'),
    color='species:N'
).properties(
    height=300,
    width=100
)

chart1 | chart2



# zoom close price
zm_date_close

zm_price_chart = alt.Chart(zm_date_close).mark_line().encode(
    x='Date',
    y='Close'
).properties( 
    height=700,
    width=700
    ).interactive()


zm_price_chart


# daily positive sentiment

# load the df for tweets 




import altair as alt
import pandas as pd

# Create a sample DataFrame
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 3, 7, 2, 6]})

# Create an Altair Chart from the DataFrame
chart = alt.Chart(data).mark_line().encode(
    x='x',
    y='y'
)

# Display the chart
chart



zm_tweet_sentiment_df = pd.read_csv('full300k_w_r_latest_sentiment_no_clean_no_finetuning.csv')
# zm_tweet_sentiment_df[['date','sentiment_r_latest']]

zm_tweet_sentiment_df['date'] = pd.to_datetime(zm_tweet_sentiment_df['date'])

daily_sum = zm_tweet_sentiment_df.groupby([zm_tweet_sentiment_df['date'].dt.date, 'sentiment_r_latest'])['sentiment_r_latest'].count()

daily_sum = daily_sum.unstack(fill_value=0)
daily_sum.reset_index(level=0, inplace=True) # date column was not flat and cannot be accessed - this flattens all columns
# st.write(daily_sum.columns)
daily_sum



import altair as alt
import pandas as pd



# Convert the 'date' column to string format for Altair
# daily_sum['date'] = daily_sum['date'].astype(str)

# Create an Altair Chart from the DataFrame
zm_positive_chart = alt.Chart(daily_sum).mark_line(color='green').encode(
    x='date',
    # y='negative:Q',
    # y2='neutral:Q',
    y='positive',
).properties(
    width=1200,
    height=400
).interactive()


zm_negative_chart = alt.Chart(daily_sum).mark_line(color='red').encode(
    x='date',
    y='negative',
    # y2='neutral:Q',
    # y='positive',
).properties(
    width=1200,
    height=400
).interactive()

zm_neutral_chart = alt.Chart(daily_sum).mark_line(color='grey').encode(
    x='date',
    # y='negative',
    y='neutral',
).properties(
    width=1200,
    height=400
).interactive()


zm_positive_chart + zm_negative_chart + zm_neutral_chart
