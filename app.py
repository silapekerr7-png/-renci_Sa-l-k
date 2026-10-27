import streamlit as st
import pandas as pd
import joblib

model = joblib.load('health_model.pkl')
df = pd.read_csv('ornek.csv')

st.title(':heart: Öğrenci Sağlık Riski Tahmini')

col1, col2 = st.columns(2)

with col1:
    sleep = st.number_input('Uyku süresi (saat)', value=float(df['sleep_duration'].median()))
    hr = st.number_input('Nabız', value=float(df['heart_rate'].median()))
    bmi = st.number_input('BMI', value=float(df['bmi'].median()))
    cal = st.number_input('Harcanan kalori', value=float(df['calorie_expenditure'].median()))
    step = st.number_input('Adım sayısı', value=float(df['step_count'].median()))
    exe = st.number_input('Egzersiz süresi (dk)', value=float(df['exercise_duration'].median()))
    water = st.number_input('Su tüketimi (litre)', value=float(df['water_intake'].median()))

with col2:
    diet = st.selectbox('Beslenme tipi', sorted(df['diet_type'].dropna().unique()))
    stress = st.selectbox('Stres seviyesi', sorted(df['stress_level'].dropna().unique()))
    quality = st.selectbox('Uyku kalitesi', sorted(df['sleep_quality'].dropna().unique()))
    activity = st.selectbox('Fiziksel aktivite', sorted(df['physical_activity_level'].dropna().unique()))
    smoke = st.selectbox('Sigara / alkol', sorted(df['smoking_alcohol'].dropna().unique()))
    gender = st.radio('Cinsiyet', sorted(df['gender'].dropna().unique()))

if st.button('Tahmin Et'):
    veri = pd.DataFrame([{'sleep_duration': sleep, 'heart_rate': hr, 'bmi': bmi, 'calorie_expenditure': cal,
                          'step_count': step, 'exercise_duration': exe, 'water_intake': water, 'diet_type': diet,
                          'stress_level': stress, 'sleep_quality': quality, 'physical_activity_level': activity,
                          'smoking_alcohol': smoke, 'gender': gender}])
    tahmin = model.predict(veri)[0]
    st.success(f"Sağlık durumu: {['fit', 'at-risk', 'unhealthy'][tahmin]}")
