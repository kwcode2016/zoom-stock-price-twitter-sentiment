
import pandas as pd

in_tweet_file = 'zm-tweet-2020-2-26-to-2020-2-29.csv'


tweet_df  = pd.read_csv(in_tweet_file)


# print(tweet_df['content'][0:30])


tweet_content_list = []

for i in tweet_df['content']:
    tweet_content_list.append(i)


print(len(tweet_content_list))


for i in tweet_content_list:
    print (i)
    input('press enter key...\n\n')


