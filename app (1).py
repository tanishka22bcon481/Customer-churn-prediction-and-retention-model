#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib


# In[3]:


#used joblib to load the trained random foresg model
model = joblib.load('churn_model.pk1')


# In[4]:


st.title("Customer Churn Prediction Dashboard")
st.write("Enter Customer Details to Predict Churn:")


# In[8]:


#input section- user selects customer details via sliders and dropdowns.
tenure= st.slider('Tenure (months)', 0, 72, 12)
monthly_charges= st.slider('Monthly Charges', 0, 150, 70)
contract = st.selectbox('Contract Type', ['month to month', 'one year', 'two year'])

#convert contract type to numerical value
contract_map= {'month to month': 0, 'one year': 1, 'two year': 2}
contract = contract_map[contract]


# In[10]:


#creating a dataframe of user input for model to  make predictions
sample = pd.DataFrame([[tenure, monthly_charges, contract]], columns=['Tenure', 'MonthlyCharges', 'Contract'])


# In[11]:


#ADDING A BUTTON- when pressed:if showed 1-churned, showed 0- not churned
if st.button('predict churn'):
    prediction = model.predict(sample)[0]
    if prediction == 1:
        st.error("this customer is likely to CHURN.")
    else:
        st.success("this customer is likely to stay.")


# In[ ]:




