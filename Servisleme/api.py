#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import joblib

# RandomForestRegressor modelini yükleme
model = joblib.load('random_forest_model.joblib')

# Tahmin fonksiyonunu tanımlama
def predict_price(model, df):
    try:
        prediction = model.predict(df)
        return prediction[0]
    except AttributeError:
        st.error("Modeliniz 'predict' fonksiyonuna sahip değil. Lütfen doğru model tipini kullanarak eğittiğinizden emin olun.")
        return None

st.title("Kadıköy ve civarındaki otellerin ortalama fiyatları")

puan_dict ={3.4: 0,
  3.9: 1,
  4.1: 2,
  4.2: 3,
  4.3: 4,
  4.4: 5,
  4.5: 6,
  4.7: 7,
  4.8: 8,
  4.9: 9,
  5.0: 10,
  5.1: 11,
  5.2: 12,
  5.3: 13,
  5.5: 14,
  5.6: 15,
  5.7: 16,
  5.8: 17,
  5.9: 18,
  6.0: 19,
  6.1: 20,
  6.2: 21,
  6.3: 22,
  6.4: 23,
  6.5: 24,
  6.6: 25,
  6.7: 26,
  6.8: 27,
  6.9: 28,
  7.0: 29,
  7.1: 30,
  7.2: 31,
  7.3: 32,
  7.4: 33,
  7.5: 34,
  7.6: 35,
  7.7: 36,
  7.8: 37,
  7.9: 38,
  8.0: 39,
  8.1: 40,
  8.2: 41,
  8.3: 42,
  8.4: 43,
  8.5: 44,
  8.6: 45,
  8.7: 46,
  8.8: 47,
  8.9: 48,
  9.0: 49,
  9.1: 50,
  9.2: 51,
  9.3: 52,
  9.4: 53,
  9.5: 54,
  9.6: 55,
  9.7: 56,
  9.8: 57,
  9.9: 58,
  10.0: 59} 
uzaklık_dict ={'Kadıköy 10.0 km uzaklıkta': 0,
  'Kadıköy 10.1 km uzaklıkta': 1,
  'Kadıköy 10.2 km uzaklıkta': 2,
  'Kadıköy 10.3 km uzaklıkta': 3,
  'Kadıköy 10.4 km uzaklıkta': 4,
  'Kadıköy 10.5 km uzaklıkta': 5,
  'Kadıköy 10.6 km uzaklıkta': 6,
  'Kadıköy 10.7 km uzaklıkta': 7,
  'Kadıköy 10.8 km uzaklıkta': 8,
  'Kadıköy 10.9 km uzaklıkta': 9,
  'Kadıköy 11.0 km uzaklıkta': 10,
  'Kadıköy 11.1 km uzaklıkta': 11,
  'Kadıköy 11.2 km uzaklıkta': 12,
  'Kadıköy 11.3 km uzaklıkta': 13,
  'Kadıköy 11.4 km uzaklıkta': 14,
  'Kadıköy 11.5 km uzaklıkta': 15,
  'Kadıköy 11.6 km uzaklıkta': 16,
  'Kadıköy 11.7 km uzaklıkta': 17,
  'Kadıköy 11.8 km uzaklıkta': 18,
  'Kadıköy 11.9 km uzaklıkta': 19,
  'Kadıköy 12.0 km uzaklıkta': 20,
  'Kadıköy 12.1 km uzaklıkta': 21,
  'Kadıköy 12.2 km uzaklıkta': 22,
  'Kadıköy 12.4 km uzaklıkta': 23,
  'Kadıköy 12.5 km uzaklıkta': 24,
  'Kadıköy 12.6 km uzaklıkta': 25,
  'Kadıköy 12.8 km uzaklıkta': 26,
  'Kadıköy 12.9 km uzaklıkta': 27,
  'Kadıköy 13.2 km uzaklıkta': 28,
  'Kadıköy 13.3 km uzaklıkta': 29,
  'Kadıköy 13.4 km uzaklıkta': 30,
  'Kadıköy 2.2 km uzaklıkta': 31,
  'Kadıköy 2.8 km uzaklıkta': 32,
  'Kadıköy 2.9 km uzaklıkta': 33,
  'Kadıköy 3.0 km uzaklıkta': 34,
  'Kadıköy 3.5 km uzaklıkta': 35,
  'Kadıköy 3.6 km uzaklıkta': 36,
  'Kadıköy 4.1 km uzaklıkta': 37,
  'Kadıköy 4.2 km uzaklıkta': 38,
  'Kadıköy 4.3 km uzaklıkta': 39,
  'Kadıköy 4.4 km uzaklıkta': 40,
  'Kadıköy 4.5 km uzaklıkta': 41,
  'Kadıköy 4.7 km uzaklıkta': 42,
  'Kadıköy 4.8 km uzaklıkta': 43,
  'Kadıköy 4.9 km uzaklıkta': 44,
  'Kadıköy 5.0 km uzaklıkta': 45,
  'Kadıköy 5.1 km uzaklıkta': 46,
  'Kadıköy 5.2 km uzaklıkta': 47,
  'Kadıköy 5.5 km uzaklıkta': 48,
  'Kadıköy 5.6 km uzaklıkta': 49,
  'Kadıköy 5.8 km uzaklıkta': 50,
  'Kadıköy 5.9 km uzaklıkta': 51,
  'Kadıköy 6.0 km uzaklıkta': 52,
  'Kadıköy 6.2 km uzaklıkta': 53,
  'Kadıköy 6.5 km uzaklıkta': 54,
  'Kadıköy 6.6 km uzaklıkta': 55,
  'Kadıköy 6.7 km uzaklıkta': 56,
  'Kadıköy 7.2 km uzaklıkta': 57,
  'Kadıköy 7.5 km uzaklıkta': 58,
  'Kadıköy 7.9 km uzaklıkta': 59,
  'Kadıköy 8.4 km uzaklıkta': 60,
  'Kadıköy 8.6 km uzaklıkta': 61,
  'Kadıköy 8.8 km uzaklıkta': 62,
  'Kadıköy 8.9 km uzaklıkta': 63,
  'Kadıköy 9.0 km uzaklıkta': 64,
  'Kadıköy 9.1 km uzaklıkta': 65,
  'Kadıköy 9.2 km uzaklıkta': 66,
  'Kadıköy 9.3 km uzaklıkta': 67,
  'Kadıköy 9.4 km uzaklıkta': 68,
  'Kadıköy 9.5 km uzaklıkta': 69,
  'Kadıköy 9.6 km uzaklıkta': 70,
  'Kadıköy 9.7 km uzaklıkta': 71,
  'Kadıköy 9.8 km uzaklıkta': 72,
  'Kadıköy 9.9 km uzaklıkta': 73}

puan = puan_dict[st.sidebar.selectbox("puan", puan_dict.keys())]
uzaklık = uzaklık_dict[st.sidebar.selectbox("uzaklık", uzaklık_dict.keys())]

features = {'puan': puan, 'uzaklık': uzaklık}
features_df = pd.DataFrame([features])
st.table(features_df)

if st.button("tahmin et"):
    pred = predict_price(model, features_df.values)
    st.write(f"Tahmin edilen değer: {pred}")


# In[ ]:




