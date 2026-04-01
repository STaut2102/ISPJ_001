import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- โหลด Data และ Model ---
@st.cache_resource
def load_assets():
    with open('ensemble_model.pkl', 'rb') as f: 
        en_model = pickle.load(f)
    df1 = pd.read_csv('restaurant_data.csv')
    return en_model, df1

en_model, df1 = load_assets()

# สร้างแผนผังแปลงชื่ออาหารเป็นตัวเลขโดยอัตโนมัติจากข้อมูลใน CSV
cuisine_list = sorted(df1['Cuisine'].unique().tolist())
cuisine_map = {name: i for i, name in enumerate(cuisine_list)}

# --- ส่วนของ Sidebar เมนู ---
st.sidebar.title("🍔 Restaurant AI")
page = st.sidebar.selectbox("เลือกหน้า", ["📊 Data Insight", "🔮 Predict Success"])

# --- หน้าที่ 1: Data Insight ---
if page == "📊 Data Insight":
    st.title("📊 Machine Learning Insight")
    st.subheader("ภาพรวมข้อมูลร้านอาหาร")
    st.bar_chart(df1['Cuisine'].value_counts())
    st.write("ราคาเฉลี่ยแยกตามประเภทอาหาร")
    st.line_chart(df1.groupby('Cuisine')['Avg_Price'].mean())

# --- หน้าที่ 2: Predict Success ---
elif page == "🔮 Predict Success":
    st.title("🔮 พยากรณ์โอกาสสำเร็จของร้าน")
    st.info("ระบุข้อมูลร้านของคุณเพื่อดูแนวโน้มการทำธุรกิจ")
    
    col1, col2 = st.columns(2)
    with col1:
        u_cuisine = st.selectbox("เลือกประเภทอาหาร", cuisine_list)
        u_price = st.number_input("ราคาเฉลี่ยต่อหัว (บาท)", min_value=0, value=200)
    with col2:
        u_score = st.slider("คะแนนทำเล (1-10)", 0.0, 10.0, 5.0)
        u_checkin = st.number_input("ยอด Social Check-in", min_value=0, value=100)
    
    if st.button("เริ่มทำการพยากรณ์"):
        # แปลงชื่ออาหารเป็นตัวเลขโดยใช้ Map ที่เราสร้างไว้ข้างบน
        c_encoded = cuisine_map[u_cuisine]
        input_data = np.array([[c_encoded, u_price, u_score, u_checkin]])
        
        prediction = en_model.predict(input_data)
        
        st.divider()
        if prediction[0] == 1:
            st.success("🎉 โอกาสรุ่งสูงมาก!")
            st.balloons()
        else:
            st.error("⚠️ มีความเสี่ยง อาจต้องปรับกลยุทธ์")
