import streamlit as st
import requests

st.set_page_config(page_title="SBD AI Balance", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ SBD AI 스트렝스 밸런스 진단")
st.caption("108만 명의 빅데이터가 분석하는 나의 숨겨진 스트렝스 포텐셜")

# 💡 [UI 보정] 슬라이더를 제거하고 정교한 수치 입력을 위해 number_input 및 selectbox로 전면 교체
with st.sidebar:
    st.header("👤 신체 스펙 입력")
    sex = st.radio("성별", ["M", "F"], horizontal=True)
    # 나이와 체중을 입력 폼으로 변경하여 소수점 및 정수 입력 편의성 극대화
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
        payload = {
            "Sex": sex, "Age": age, "BodyweightKg": weight, "Equipment": equipment,
            "Known_Type": known_type, "Known_1RM": known_1rm, "Target_Type": target_type,
            "Target_Actual_1RM": target_actual if target_actual > 0 else None
        }
        
        try:
            # FastAPI 서버와 실시간 비동기 통신
            res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()
            
            st.write("---")
            # 결과를 직관적인 대시보드 메트릭 카드로 출력
            c1, c2, c3 = st.columns(3)
            c1.metric(label="AI 권장 중량", value=f"{res['predicted_target_1rm']} kg")
            c2.metric(label="당일 컨디션 최상 (상한선)", value=f"{res['confidence_interval']['upper_kg']} kg")
            
            if res['balance_gap_kg'] is not None:
                # 갭 차이 시각화 (유저가 더 잘 치면 초록색 우상향, 부족하면 빨간색 하향)
                delta_val = -res['balance_gap_kg']
                c3.metric(label="표준 대비 갭 (Gap)", value=f"{res['balance_gap_kg']} kg", delta=f"{delta_val:.1f} kg")
                
            st.write("")
            # 교정된 고도화 팩폭 멘트 출력
            st.text(res["message"])
            
        except Exception as e:
            st.error(f"서버 연결 실패. FastAPI 가동 상태를 확인하세요. 에러: {e}")

with tab2:
    st.subheader("🔮 만약에... 내가 몸무게를 더 늘리거나 줄인다면?")
    st.info("사이드바의 체중(kg)과 나이를 조작한 뒤 다시 [스캔 시작]을 눌러보세요. 바뀐 스펙에 맞춰 AI가 미래의 잠재 중량을 실시간으로 재계산합니다.")