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
    # ตรวจสอบว่ามีไฟล์ ensemble_model.pkl และ restaurant_data.csv ใน GitHub แล้ว
    with open('ensemble_model.pkl', 'rb') as f: 
        en_model = pickle.load(f)
    df = pd.read_csv('restaurant_data.csv')
    
    # --- Data Cleaning ป้องกัน Error ---
    df['Cuisine'] = df['Cuisine'].fillna('Other').astype(str)
    # บรรทัดนี้สำคัญ: ป้องกัน TypeError จากค่าราคาที่ไม่ใช่ตัวเลข
    df['Avg_Price'] = pd.to_numeric(df['Avg_Price'], errors='coerce').fillna(0)
    
    return en_model, df

en_model, df1 = load_assets()

# เตรียมข้อมูลสำหรับ Selectbox
cuisine_list = sorted(df1['Cuisine'].unique().tolist())
cuisine_map = {name: i for i, name in enumerate(cuisine_list)}

# --- หน้าจอหลัก ---
st.set_page_config(page_title="AI Restaurant SUCCESS", layout="centered")
st.title("🍔 AI พยากรณ์ความสำเร็จร้านอาหาร")

menu = st.sidebar.selectbox("เลือกเมนู", ["Dashboard ข้อมูล", "ทำนายโอกาสรอด"])

if menu == "Dashboard ข้อมูล":
    st.header("📊 ข้อมูลร้านอาหารในระบบ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**จำนวนร้านแยกตามประเภท**")
        st.bar_chart(df1['Cuisine'].value_counts())
    with col2:
        st.write("**ราคาเฉลี่ยต่อจาน (บาท)**")
        # คำนวณค่าเฉลี่ยเฉพาะคอลัมน์ที่เป็นตัวเลข
        avg_price_chart = df1.groupby('Cuisine')['Avg_Price'].mean()
        st.line_chart(avg_price_chart)

else:
    st.header("🔮 ลองทำนายธุรกิจของคุณ")
    u_c = st.selectbox("ประเภทอาหาร", cuisine_list)
    u_p = st.number_input("ราคาสินค้าเฉลี่ย (บาท)", min_value=0, value=150)
    u_s = st.slider("คะแนนทำเล (1-10)", 0.0, 10.0, 7.0)
    u_ch = st.number_input("ยอด Check-in โซเชียล", min_value=0, value=500)
    
    if st.button("ประมวลผลด้วย AI", use_container_width=True):
        # แปลง Input เป็นตัวเลขตาม Map ที่สร้างไว้เพื่อให้ Model อ่านได้
        input_data = np.array([[cuisine_map[u_c], u_p, u_s, u_ch]])
        res = en_model.predict(input_data)
        
        st.divider()
        if res[0] == 1:
            st.success("🎉 รุ่งแน่แม่จ๋า")
            st.balloons()
        else:
            st.error("⚠️ เสี่ยงไปหน่อย อย่าปล่อยผ่าน")
