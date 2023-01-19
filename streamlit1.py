import streamlit as st
import pandas as pd

st.write('''
# My first app

Hello *World!*

this is another line


this is a short paragraph to see if it will update after changes


nice! there is an option that allows it to always rerun after changes in source code.

this will be very helpful in the future. I can use streamlit for future personal projects too.

another test

testing

''')


txt = st.text_area("test text area")

# note that text will automatically update after new text is added. very nice.
st.write("txt is: ")
st.write(txt)




import datetime
date = st.sidebar.date_input('start date', datetime.date(2022,1,1))


st.write(date)






