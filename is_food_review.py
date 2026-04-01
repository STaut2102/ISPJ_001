import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Restaurant AI", layout="wide")

# --- โหลด Assets (Cache ไว้เพื่อความเร็ว) ---
@st.cache_resource
def load_assets():
    # โหลดไฟล์จากขั้นตอนที่เราเตรียมใน Colab
    with open('model_ml.pkl', 'rb') as f: ml = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: sc = pickle.load(f)
    with open('le_cuisine.pkl', 'rb') as f: le = pickle.load(f)
    nn = load_model('model_nn.h5')
    df = pd.read_csv('data_restaurant.csv')
    
    # Clean ข้อมูลเพื่อป้องกัน Error แบบในรูปที่ 2
    df['Cuisine'] = df['Cuisine'].fillna('Other').astype(str)
    df['Avg_Price'] = pd.to_numeric(df['Avg_Price'], errors='coerce').fillna(0)
    
    return ml, sc, le, nn, df

# เรียกใช้ฟังก์ชันโหลดข้อมูล
try:
    ml_model, scaler, le_cuisine, nn_model, df = load_assets()
except Exception as e:
    st.error(f"กรุณาตรวจสอบว่าอัปโหลดไฟล์ model_ml.pkl, model_nn.h5, scaler.pkl, le_cuisine.pkl และ data_restaurant.csv ครบแล้ว")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🍱 Restaurant AI")
menu = st.sidebar.radio("เลือกหน้า", ["Info: ML Theory", "Info: NN Theory", "Test: ML Predict", "Test: NN Predict"])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Designed with support from **Gemini AI (Google)**")

# --- หน้า 1: อธิบาย ML ---
if menu == "Info: ML Theory":
    st.title("📊 ทฤษฎี Machine Learning")
    st.write("โมเดลนี้ใช้ **Ensemble Learning (Soft Voting)** โดยรวมพลังจาก RF, XGBoost และ Logistic Regression เข้าด้วยกัน")
    st.write("มีการใช้ **StandardScaler** เพื่อปรับสมดุลข้อมูล (Scaling) ทำให้ปัจจัยราคาและทำเลมีความสำคัญเท่าเทียมกับยอดเช็คอิน")

# --- หน้า 2: อธิบาย NN ---
elif menu == "Info: NN Theory":
    st.title("🧠 ทฤษฎี Neural Network")
    st.write("โมเดลนี้ใช้โครงสร้าง **ANN (Artificial Neural Network)** 4 ชั้น")
    st.write("ใช้ **Dropout Layer** เพื่อป้องกันการจำคำตอบ และ **Sigmoid** เพื่อคำนวณโอกาสรอดเป็นเปอร์เซ็นต์")

# --- หน้า 3: ทดสอบ ML ---
elif menu == "Test: ML Predict":
    st.title("🔮 ทดสอบด้วย Machine Learning")
    c = st.selectbox("ประเภทอาหาร", le_cuisine.classes_)
    p = st.number_input("ราคาเฉลี่ยต่อจาน", value=150)
    l = st.slider("ทำเล (1-10)", 1, 10, 5)
    s = st.number_input("ยอด Check-in", value=100)
    
    if st.button("ประมวลผล (ML)"):
        input_data = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        input_scaled = scaler.transform(input_data)
        prob = ml_model.predict_proba(input_scaled)[0][1] * 100
        st.subheader(f"โอกาสรอด: {prob:.2f}%")
        st.progress(prob/100)

# --- หน้า 4: ทดสอบ NN ---
elif menu == "Test: NN Predict":
    st.title("🤖 ทดสอบด้วย Neural Network")
    c = st.selectbox("ประเภทอาหาร ", le_cuisine.classes_)
    p = st.number_input("ราคาต่อหัว ", value=150)
    l = st.slider("ทำเลคะแนน ", 1, 10, 5)
    s = st.number_input("เช็คอินโซเชียล ", value=100)
    
    if st.button("ประมวลผล (NN)"):
        input_data = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        input_scaled = scaler.transform(input_data)
        prob = float(nn_model.predict(input_scaled)[0][0]) * 100
        st.subheader(f"โอกาสรอด: {prob:.2f}%")
        st.progress(prob/100)
