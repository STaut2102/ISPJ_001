import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. โหลดข้อมูลจากขั้นตอนที่ 1
df = pd.read_csv('data_restaurant.csv')

# 2. กระบวนการเตรียมข้อมูล (Data Preparation)
le = LabelEncoder()
df['Cuisine_Enc'] = le.fit_transform(df['Cuisine'])

# เลือก Features หลัก: ประเภทอาหาร, ราคา, ทำเล, เช็คอิน
X = df[['Cuisine_Enc', 'Avg_Price', 'Location_Score', 'Social_Checkin']]
y = df['Status']

# --- การทำ Scaling (หัวใจสำคัญของความเสถียร) ---
# วิธีนี้จะปรับให้ตัวเลขหลักหน่วย (ทำเล) และหลักพัน (เช็คอิน) มีน้ำหนักเท่ากันในสายตา AI
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. พัฒนาโมเดล 1: Machine Learning (Ensemble - Soft Voting)
# ผสม 3 เทคนิคเพื่อให้ได้ความแม่นยำสูงและบอกเป็น % ได้
m1 = RandomForestClassifier(n_estimators=100, random_state=42)
m2 = XGBClassifier(eval_metric='logloss', random_state=42)
m3 = LogisticRegression(random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('rf', m1), ('xgb', m2), ('lr', m3)], 
    voting='soft' # ใช้ soft เพื่อให้คำนวณออกมาเป็นความน่าจะเป็น (%)
)
ensemble_model.fit(X_scaled, y)

# 4. พัฒนาโมเดล 2: Neural Network (ANN)
# ออกแบบโครงสร้างเอง: 4 Input -> 16 Neurons -> 8 Neurons -> 1 Output
model_nn = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dropout(0.2), # ลดความเสี่ยงในการจำคำตอบ (Overfitting)
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') # ทำนายผลออกมาเป็นค่าระหว่าง 0 ถึง 1
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(X_scaled, y, epochs=50, verbose=0)

# 5. บันทึกไฟล์ "หัวใจ" ทั้งหมดเพื่อนำไปอัปโหลดขึ้น GitHub
with open('model_ml.pkl', 'wb') as f: pickle.dump(ensemble_model, f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('le_cuisine.pkl', 'wb') as f: pickle.dump(le, f)
model_nn.save('model_nn.h5')

print("✅ เตรียมข้อมูลและสร้างโมเดลทั้ง 2 ประเภท (ML & NN) เรียบร้อยแล้ว!")
