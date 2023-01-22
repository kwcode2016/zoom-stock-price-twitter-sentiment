# quick look at tweets from different dates
tweet_json = 'zm-6-2019-04-01-2022-12-11-all-zm-tweets.jsonl'
zm_price_filename = 'zoom-stock-prices-2019-04-01-2022-12-11.csv'
# output_csv = 'zm-tweets-jan to april 2020.csv'

def breakpoint(message = 'breakpoint reached'):
    import sys
    sys.exit(message)


import pandas as pd


print(tweet_json)
print(zm_price_filename)

from_date = '2020-9-11'
to_date = '2020-9-13'



output_name = f'zm-tweet-{from_date}-to-{to_date}.csv'

print(output_name)

# breakpoint('testing message')

tweet_zm_df = pd.read_json(tweet_json, lines=True)

jan_to_april_2020_tweet_df = tweet_zm_df[(tweet_zm_df['date'] > from_date) & (tweet_zm_df['date'] <= to_date)]

jan_to_april_2020_tweet_df.to_csv(output_name)


# load csv

# batch_tweets = pd.read_csv(in_csv)

# print(batch_tweets.info())


# for i in batch_tweets[0:10]:
#     print (i)   





print('Complete.')

