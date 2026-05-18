import streamlit as st
import pandas as pd
import joblib

# 형님이 올리신 setup.py의 함수들을 가져옵니다.
from setup import apply_preprocessing, strategy_map

# 페이지 기본 세팅
st.set_page_config(page_title="SBD AI Balance", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ SBD AI 스트렝스 밸런스 진단")
st.caption("108만 명의 빅데이터가 분석하는 나의 숨겨진 스트렝스 포텐셜")

# 아티팩트 보따리 로드
@st.cache_resource
def load_ai_model():
    try:
        artifact = joblib.load('SBD_v2.pkl')
        return artifact
    except Exception as e:
        st.error(f"🚨 아티팩트 로드 실패: {e}")
        return None

artifact = load_ai_model()

# 사이드바 신체 스펙 입력 영역
with st.sidebar:
    st.header("👤 신체 스펙 입력")
    sex = st.radio("성별", ["M", "F"], horizontal=True)
    age = st.number_input("나이 (만)", min_value=17, max_value=100, value=30, step=1)
    weight = st.number_input("현재 체중 (kg)", min_value=30.0, max_value=300.0, value=101.0, step=0.1)
    equipment = st.selectbox("착용 장비", ["Raw", "Wraps", "Single-ply"])

# 메인 화면 탭 구조
tab1, tab2 = st.tabs(["📊 황금 비율 진단", "🔮 What-If 벌크업 시뮬레이터"])

with tab1:
    st.subheader("내 진짜 스트렝스 밸런스는?")
    col1, col2 = st.columns(2)
    with col1:
        known_type = st.selectbox("가장 자신 있는 종목 (기준)", ["Squat", "Bench", "Deadlift"])
        known_1rm = st.number_input("해당 종목 현재 1RM (kg)", min_value=0.0, max_value=600.0, value=190.0, step=2.5)
    with col2:
        target_type = st.selectbox("AI 진단 타겟 종목", ["Bench", "Deadlift", "Squat"])
        target_actual = st.number_input("내 진짜 타겟 종목 기록 (kg)", min_value=0.0, max_value=600.0, value=130.0, step=2.5)
        
    st.write("")
    if st.button("바벨 황금 밸런스 스캔 시작", use_container_width=True):
        if artifact is None:
            st.error("AI 모델 로드 실패 상태입니다.")
        else:
            model = artifact['model']
            saved_preprocessor = artifact['preprocessor']
            mae_value = artifact['mae']
            
            # 1. 입력 데이터 딕셔너리 생성
            input_dict = {
                "Sex": sex, "Age": age, "BodyweightKg": weight, "Equipment": equipment,
                "Known_Type": known_type, "Known_1RM": known_1rm, "Target_Type": target_type,
                "Target_Actual_1RM": target_actual if target_actual > 0 else None
            }
            
            try:
                # 2. 판다스 데이터프레임 변환 및 상류 파생변수 생성
                input_df = pd.DataFrame([input_dict])
                input_df['Age_From_Peak'] = abs(input_df['Age'] - 32.0)
                input_df['Input_Rel_Strength'] = input_df['Known_1RM'] / input_df['BodyweightKg']
                
                # 3. 형님의 순정 전처리 엔진 통과
                processed_df, _ = apply_preprocessing(
                    X_input=input_df,
                    strategy_map=strategy_map,
                    is_train=False,
                    preprocessor=saved_preprocessor
                )
                
                # 4. 강제 컬럼 이름 복구 치트키
                if not isinstance(processed_df, pd.DataFrame):
                    processed_df = pd.DataFrame(processed_df)
                processed_df.columns = model.feature_names_
                
                # 5. 캣부스트 마스터 모델 추론 및 신뢰구간 계산
                predicted_target = float(model.predict(processed_df)[0])
                MAE_VALUE = mae_value 
                
                lower_bound = predicted_target - MAE_VALUE
                upper_bound = predicted_target + MAE_VALUE
                
                # 6. 밸런스 갭(Gap) 연산
                balance_gap = None
                gap_text = "현재 타
