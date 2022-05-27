import streamlit as st
import helper
import pickle

model = pickle.load(open('model.pkl', 'rb')) #unpickel the model

st.header('Duplicate question pairs') #webpage with the header

q1 = st.text_input('Enter question 1') #text inputs
q2 = st.text_input('Enter question 2')

if st.button('Find'): #button 'find'
    query = helper.query_point_creator(q1, q2) #call the query point creator
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')