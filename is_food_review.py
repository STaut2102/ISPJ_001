import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Restaurant AI Success", layout="wide")

# --- โหลด Assets ---
@st.cache_resource
def load_all():
    with open('model_ml.pkl', 'rb') as f: ml = pickle.load(f)
    with open('model_nn.pkl', 'rb') as f: nn = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: sc = pickle.load(f)
    with open('le_cuisine.pkl', 'rb') as f: le = pickle.load(f)
    df = pd.read_csv('data_restaurant.csv')
    return ml, nn, sc, le, df

try:
    ml_model, nn_model, scaler, le_cuisine, df_raw = load_all()
except:
    st.error("ไม่พบไฟล์โมเดล กรุณาอัปโหลดไฟล์ .pkl และ .csv ให้ครบถ้วน")
    st.stop()

# --- Sidebar ---
st.sidebar.title("📑 เมนูหลัก")
page = st.sidebar.radio("เลือกหน้า", ["Info: ML Model", "Info: NN Model", "Test: ML Predict", "Test: NN Predict"])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Datasets & Model Architecture designed with support from **Gemini AI (Google)**")

# --- หน้า 1: อธิบาย ML ---
if page == "Info: ML Model":
    st.title("📊 Machine Learning Model Information")
    st.write("### แนวทางการพัฒนา (Ensemble Learning)")
    st.write("ใช้การรวมพลังของ RandomForest, XGBoost และ Logistic Regression ด้วยวิธี Soft Voting เพื่อความเสถียร")
    st.info("ใช้ StandardScaler เพื่อปรับสมดุลข้อมูล ลดการพึ่งพาตัวแปรใดตัวแปรหนึ่งมากเกินไป")

# --- หน้า 2: อธิบาย NN ---
elif page == "Info: NN Model":
    st.title("🧠 Neural Network Model Information")
    st.write("### โครงสร้างโมเดล (Multi-layer Perceptron)")
    st.write("ใช้ MLPClassifier จาก Scikit-learn ในการสร้างโครงสร้างประสาทเทียม 2 ชั้นซ่อน (Hidden Layers: 16, 8)")
    st.write("เน้นความรวดเร็วและแม่นยำในการทำนายผลลัพธ์แบบ Binary Classification")

# --- หน้า 3: ทดสอบ ML ---
elif page == "Test: ML Predict":
    st.title("🔮 พยากรณ์โอกาสสำเร็จ (Machine Learning)")
    c = st.selectbox("ประเภทอาหาร", le_cuisine.classes_)
    p = st.number_input("ราคาเฉลี่ยต่อจาน (บาท)", value=150)
    l = st.slider("คะแนนทำเล (1-10)", 1, 10, 5)
    s = st.number_input("ยอดเช็คอินโซเชียล", value=100)
    
    if st.button("เริ่มคำนวณโอกาสรอด (ML)", use_container_width=True):
        input_data = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        scaled_input = scaler.transform(input_data)
        prob = ml_model.predict_proba(scaled_input)[0][1] * 100
        
        st.subheader(f"โอกาสรอด: {prob:.2f}%")
        st.progress(prob/100)
        if prob >= 50: st.success("สถานะ: รอด")
        else: st.error("สถานะ: ร่วง")

# --- หน้า 4: ทดสอบ NN ---
elif page == "Test: NN Predict":
    st.title("🤖 พยากรณ์โอกาสสำเร็จ (Neural Network)")
    c = st.selectbox("เลือกประเภทอาหาร", le_cuisine.classes_)
    p = st.number_input("ระบุราคาเฉลี่ย", value=150)
    l = st.slider("ระบุคะแนนทำเล", 1, 10, 5)
    s = st.number_input("ระบุยอดเช็คอิน", value=100)
    
    if st.button("เริ่มคำนวณโอกาสรอด (NN)", use_container_width=True):
        input_data = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        scaled_input = scaler.transform(input_data)
        prob = nn_model.predict_proba(scaled_input)[0][1] * 100
        
        st.subheader(f"โอกาสรอด (NN ประมวลผล): {prob:.2f}%")
        st.progress(prob/100)
