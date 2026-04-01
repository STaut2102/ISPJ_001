import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- โหลดไฟล์ Assets ---
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
    st.error("❌ พบปัญหาการโหลดไฟล์: กรุณาตรวจสอบว่ามีไฟล์ .pkl และ .csv ครบถ้วนบน GitHub")
    st.stop()

# --- Sidebar พร้อม Credit ---
st.sidebar.title("📑 AI Restaurant Project")
page = st.sidebar.radio("เมนูหลัก", ["Info: ML Theory", "Info: NN Theory", "Test: ML Predict", "Test: NN Predict"])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Datasets & Architecture designed with **Gemini AI (Google)**")

# --- หน้า 1: อธิบาย ML ---
if page == "Info: ML Theory":
    st.title("📊 ทฤษฎี Machine Learning")
    st.write("### แนวทางการพัฒนา (Ensemble Learning)")
    st.markdown("""
    - **อัลกอริทึม:** ใช้การรวมพลังของ RandomForest, XGBoost และ Logistic Regression
    - **การเตรียมข้อมูล:** มีการทำ **Standardization** เพื่อปรับสเกลของปัจจัยต่างๆ ให้เท่ากัน
    - **การพยากรณ์:** ใช้ระบบ **Soft Voting** เพื่อคำนวณหาความน่าจะเป็นเฉลี่ยจากทุกโมเดล
    - **แหล่งอ้างอิง:** Scikit-learn Documentation
    """)

# --- หน้า 2: อธิบาย NN ---
elif page == "Info: NN Theory":
    st.title("🧠 ทฤษฎี Neural Network")
    st.write("### แนวทางการพัฒนา (Multi-layer Perceptron)")
    st.markdown("""
    - **อัลกอริทึม:** ใช้โครงสร้างประสาทเทียมแบบ Feed-forward
    - **โครงสร้างชั้นซ่อน:** แบ่งเป็น 2 ชั้น (16 และ 8 Neurons)
    - **ฟังก์ชันกระตุ้น:** ใช้ 'ReLU' ในชั้นซ่อน และคำนวณผลลัพธ์สุดท้ายเป็นความน่าจะเป็น
    - **แหล่งอ้างอิง:** Multi-layer Perceptron Classifier by Scikit-learn
    """)

# --- หน้า 3: ทดสอบ ML ---
elif page == "Test: ML Predict":
    st.title("🔮 พยากรณ์ด้วย Machine Learning")
    c = st.selectbox("เลือกประเภทอาหาร", le_cuisine.classes_)
    p = st.number_input("ราคาเฉลี่ยต่อจาน", value=150)
    l = st.slider("ทำเล (1 = ห่วย, 10 = ดีเยี่ยม)", 1, 10, 5)
    s = st.number_input("ยอด Check-in โซเชียล", value=100)
    
    if st.button("ประมวลผลด้วย ML", use_container_width=True):
        # แก้ไขจุดที่ 4: ต้องนำข้อมูลไป Scaling ก่อนทำนาย
        inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        inp_scaled = scaler.transform(inp)
        
        prob = ml_model.predict_proba(inp_scaled)[0][1] * 100
        st.subheader(f"โอกาสความสำเร็จ: {prob:.2f}%")
        st.progress(prob/100)

# --- หน้า 4: ทดสอบ NN ---
elif page == "Test: NN Predict":
    st.title("🤖 พยากรณ์ด้วย Neural Network")
    c = st.selectbox("ประเภทอาหาร ", le_cuisine.classes_)
    p = st.number_input("ราคาต่อจาน ", value=150)
    l = st.slider("คะแนนทำเล ", 1, 10, 5)
    s = st.number_input("ยอดเช็คอิน ", value=100)
    
    if st.button("ประมวลผลด้วย NN", use_container_width=True):
        inp = np.array([[le_cuisine.transform([c])[0], p, l, s]])
        inp_scaled = scaler.transform(inp)
        
        # แก้ไขจุดที่ 5: ใช้ predict_proba กับ MLPClassifier (ไม่ต้องใช้ TensorFlow)
        prob = nn_model.predict_proba(inp_scaled)[0][1] * 100
        st.subheader(f"โอกาสความสำเร็จ (AI ประมวลผล): {prob:.2f}%")
        st.progress(prob/100)
