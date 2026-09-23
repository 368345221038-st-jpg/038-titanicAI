"""
ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิค (Titanic Survival Prediction)
พัฒนาโดย: นายวิศรุต อินโต

โมเดล: titanic_mlp.keras
Features (เรียงตามลำดับที่ใช้เทรนโมเดล):
    1. Pclass        - ชั้นโดยสาร (1, 2, 3)
    2. Sex_female     - เพศหญิง = 1, เพศชาย = 0
    3. Age            - อายุ (scaled ด้วย MinMaxScaler)
    4. Fare           - ค่าโดยสาร (scaled ด้วย MinMaxScaler)
    5. FamilySize     - ขนาดครอบครัวบนเรือ (SibSp + Parch + 1)

หมายเหตุสำคัญ: ค่า MIN/MAX ที่ใช้ scale ตัวแปร Age และ Fare ด้านล่างนี้
อิงจากค่าต่ำสุด-สูงสุดทั่วไปของชุดข้อมูล Titanic (train.csv) เนื่องจากไฟล์
.keras ที่แนบมาไม่ได้เก็บ MinMaxScaler ที่ fit ไว้ตอนเทรนจริงมาด้วย
==> ถ้าค่า min/max ที่ใช้ตอนเทรนต่างจากนี้ กรุณาแก้ไขค่าคงที่ AGE_MIN, AGE_MAX,
FARE_MIN, FARE_MAX ให้ตรงกับค่าที่ใช้ fit scaler จริงของท่าน เพื่อผลการทำนาย
ที่ถูกต้องแม่นยำ
"""

import numpy as np
import streamlit as st
from tensorflow import keras

# ------------------------------------------------------------------
# ค่าคงที่ของระบบ — แก้ไขให้ตรงกับตอนเทรนโมเดลจริงของท่าน
# ------------------------------------------------------------------
MODEL_PATH = "titanic_mlp.keras"

# ค่า min/max สำหรับ MinMaxScaler (อิงค่ามาตรฐานของชุดข้อมูล Titanic)
AGE_MIN, AGE_MAX = 0.42, 80.0
FARE_MIN, FARE_MAX = 0.0, 512.3292

# ความแม่นยำของโมเดล (แก้ไขให้ตรงกับผลประเมินจริงของท่าน เช่น จาก
# model.evaluate() หรือ accuracy_score บน test set)
MODEL_ACCURACY = 0.82  # <-- TODO: ใส่ค่าความแม่นยำจริงของโมเดลท่าน

st.set_page_config(
    page_title="ระบบทำนายการรอดชีวิตผู้โดยสารไททานิค",
    page_icon="🚢",
    layout="centered",
)

# ------------------------------------------------------------------
# CSS มินิมอล โทนสีสุภาพ
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #FAFAFA; }
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #1F2A44;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            color: #6B7280;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1rem;
        }
        .survived {
            background-color: #E6F4EA;
            border: 1px solid #34A853;
            color: #1E6B34;
        }
        .not-survived {
            background-color: #FCE8E6;
            border: 1px solid #EA4335;
            color: #A32E25;
        }
        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #E5E7EB;
        }
        div.stButton > button {
            background-color: #1F2A44;
            color: white;
            border-radius: 8px;
            width: 100%;
            padding: 0.6rem;
            font-weight: 600;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #33456B;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


model = load_model()

# ------------------------------------------------------------------
# ส่วนหัวของหน้าเว็บ
# ------------------------------------------------------------------
st.markdown('<div class="main-title">🚢 ระบบทำนายการรอดชีวิตผู้โดยสารไททานิค</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Titanic Survival Prediction System (Neural Network)</div>', unsafe_allow_html=True)

acc_col1, acc_col2, acc_col3 = st.columns([1, 1.2, 1])
with acc_col2:
    st.metric(label="ความแม่นยำของโมเดล (Model Accuracy)", value=f"{MODEL_ACCURACY * 100:.1f}%")

st.divider()

# ------------------------------------------------------------------
# ฟอร์มกรอกข้อมูลผู้โดยสาร
# ------------------------------------------------------------------
st.subheader("กรอกข้อมูลผู้โดยสาร")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", options=[1, 2, 3], index=2,
                           help="1 = ชั้นหนึ่ง, 2 = ชั้นสอง, 3 = ชั้นสาม")
    sex = st.radio("เพศ (Sex)", options=["หญิง", "ชาย"], horizontal=True)
    age = st.slider("อายุ (Age)", min_value=0, max_value=80, value=30)

with col2:
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสบนเรือ (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("จำนวนพ่อแม่/บุตรบนเรือ (Parch)", min_value=0, max_value=10, value=0)

family_size = sibsp + parch + 1
st.caption(f"ขนาดครอบครัวบนเรือ (FamilySize) ที่คำนวณได้: **{family_size}**")

predict_clicked = st.button("ทำนายผล")

# ------------------------------------------------------------------
# การประมวลผลและทำนาย
# ------------------------------------------------------------------
if predict_clicked:
    sex_female = 1 if sex == "หญิง" else 0

    age_scaled = (age - AGE_MIN) / (AGE_MAX - AGE_MIN)
    fare_scaled = (fare - FARE_MIN) / (FARE_MAX - FARE_MIN)
    age_scaled = float(np.clip(age_scaled, 0.0, 1.0))
    fare_scaled = float(np.clip(fare_scaled, 0.0, 1.0))

    features = np.array([[pclass, sex_female, age_scaled, fare_scaled, family_size]], dtype=np.float32)

    probability = float(model.predict(features, verbose=0)[0][0])
    survived = probability >= 0.5

    st.markdown("### ผลการทำนาย")
    if survived:
        st.markdown(
            f"""
            <div class="result-card survived">
                <h3>✅ มีแนวโน้มรอดชีวิต</h3>
                <p>ความน่าจะเป็นในการรอดชีวิต: <b>{probability * 100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card not-survived">
                <h3>❌ มีแนวโน้มไม่รอดชีวิต</h3>
                <p>ความน่าจะเป็นในการรอดชีวิต: <b>{probability * 100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(probability)

# ------------------------------------------------------------------
# ส่วนท้ายของหน้าเว็บ
# ------------------------------------------------------------------
st.markdown(
    '<div class="footer">พัฒนาโดย: นายวิศรุต อินโต</div>',
    unsafe_allow_html=True,
)
