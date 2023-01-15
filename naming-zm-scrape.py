import os

from_date = '2019-04-01'
to_date = '2022-12-11'
file_name_message = "all zm tweets"
file_name_message = file_name_message.replace(' ', '-')
file_number = '5'
max_results = 1000000000000000

import glob
zm_file_list = glob.glob('zm-*.jsonl')

print('\nstart..')
for f in zm_file_list:
    print (f)
print('...end\n')

print('getting largest number from zm-??-')

#try getting the last filename and getting the number
print('last file found for zm-*.jsonl:')
last_zm_file_number = zm_file_list[-1].split('-')[1]
file_number = int(last_zm_file_number) + 1




# print (file_name_message)

cli_command = f'snscrape --jsonl --progress --max-results {max_results} --since {from_date} twitter-search "\$ZM until:{to_date}" > zm-{file_number}-{from_date}-{to_date}-{file_name_message}.jsonl'

print(cli_command)


os.system('conda activate')
os.system(cli_command)