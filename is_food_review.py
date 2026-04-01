import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 

# --- โหลด Data และ Model ---
@st.cache_resource
def load_assets():
    with open('ensemble_model.pkl', 'rb') as f: 
        en_model = pickle.load(f)
    df1 = pd.read_csv('restaurant_data.csv')
    return en_model, df1

en_model, df1 = load_assets()

# แปลงชื่ออาหารเป็นเลข (Manual Mapping จากข้อมูลจริง)
cuisine_list = sorted(df1['Cuisine'].unique().tolist())
cuisine_map = {name: i for i, name in enumerate(cuisine_list)}

# --- หน้าจอหลัก ---
st.set_page_config(page_title="AI Restaurant SUCCESS", layout="centered")
st.title("🍔 AI พยากรณ์ความสำเร็จร้านอาหาร")

menu = st.sidebar.selectbox("เลือกเมนู", ["Dashboard ข้อมูล", "ทำนายโอกาสรอด"])

if menu == "Dashboard ข้อมูล":
    st.header("📊 ข้อมูลร้านอาหารในระบบ")
    st.bar_chart(df1['Cuisine'].value_counts())
    st.write("ราคาเฉลี่ยต่อจาน:")
    st.line_chart(df1.groupby('Cuisine')['Avg_Price'].mean())

else:
    st.header("🔮 ลองทำนายธุรกิจของคุณ")
    u_c = st.selectbox("ประเภทอาหาร", cuisine_list)
    u_p = st.number_input("ราคาสินค้าเฉลี่ย (บาท)", min_value=0, value=150)
    u_s = st.slider("คะแนนทำเล (1-10)", 0.0, 10.0, 7.0)
    u_ch = st.number_input("ยอด Check-in โซเชียล", min_value=0, value=500)
    
    if st.button("ประมวลผลด้วย AI"):
        input_data = np.array([[cuisine_map[u_c], u_p, u_s, u_ch]])
        res = en_model.predict(input_data)
        
        if res[0] == 1:
            st.success("🎉 ร้านนี้รุ่งชัวร์!")
            st.balloons()
        else:
            st.error("⚠️ ร้านนี้มีความเสี่ยงสูงปรับปรุงด่วน")
