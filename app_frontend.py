import streamlit as st
import pandas as pd
import joblib
import datetime
from setup import apply_preprocessing, strategy_map

# 페이지 기본 세팅
st.set_page_config(page_title="SBD AI Balance", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ SBD AI 스트렝스 밸런스 진단")
st.caption("108만 명의 빅데이터가 분석하는 나의 숨겨진 스트렝스 포텐셜")

# 💡 [백엔드 통합] 외부 FastAPI 주소를 찌르지 않고, 클라우드 메모리에 모델을 다이렉트로 로드합니다.
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
            
            # 입력 데이터 스냅샷 딕셔너리 생성
            input_dict = {
                "Sex": sex, "Age": age, "BodyweightKg": weight, "Equipment": equipment,
                "Known_Type": known_type, "Known_1RM": known_1rm, "Target_Type": target_type,
                "Target_Actual_1RM": target_actual if target_actual > 0 else None
            }
            
            try:
                # [A] 판다스 변환 및 상류 파생변수 실시간 생성
                input_df = pd.DataFrame([input_dict])
                input_df['Age_From_Peak'] = abs(input_df['Age'] - 32.0)
                input_df['Input_Rel_Strength'] = input_df['Known_1RM'] / input_df['BodyweightKg']
                
                # [B] 하류 전처리 마스터 엔진 통과
                processed_df, _ = apply_preprocessing(
                    X_input=input_df,
                    strategy_map=strategy_map,
                    is_train=False,
                    preprocessor=saved_preprocessor
                )
                
                # [C] 캣부스트 마스터 모델 추론 및 신뢰구간 순차 계산
                predicted_target = float(model.predict(processed_df)[0])
                MAE_VALUE = mae_value 
                
                lower_bound = predicted_target - MAE_VALUE
                upper_bound = predicted_target + MAE_VALUE
                
                # [D] 유저의 실제 기록이 입력되었다면 밸런스 갭(Gap) 연산
                balance_gap = None
                gap_text = "현재 타겟 종목의 실제 기록을 입력하지 않아 밸런스 갭 분석은 생략됩니다."
                
                if target_actual > 0:
                    balance_gap = predicted_target - target_actual
                    gap_status = "부족한 약점" if balance_gap > 0 else "초과 달성한 강점"
                    gap_text = f"표준 비율 대비 현재 약 {abs(balance_gap):.1f} kg [{gap_status}] 상태입니다."

                # [E] 고도화 팩폭 멘트 튜닝
                if balance_gap is not None:
                    if balance_gap > 0:
                        advice_comment = f"현재 {target_type} 성능이 정석 비율보다 밀리고 있습니다. 약점 보완 루틴 돌리세요."
                    else:
                        advice_comment = f"이미 {target_type} 성능은 체급 대비 차고 넘칩니다. {known_type}에 더 집중하셔도 좋습니다."
                else:
                    advice_comment = "타겟 종목 기록을 적어주시면 더 정밀한 코칭 멘트가 나갑니다."

                custom_message = (
                    f"--------------------------------------------------\n"
                    f"🏋️‍♂️ AI 기반 SBD 스트렝스 밸런스 진단 리포트\n"
                    f"--------------------------------------------------\n"
                    f" 체급 대비 정석적인 신체 역학 비율을 스캔한 결과입니다.\n\n"
                    f"[*] 현재 스펙 : {sex} / {age:.0f}세 / {weight:.1f}kg / {equipment}\n"
                    f"[*] 기준 기록 : {known_type} -> {known_1rm:.1f} kg\n"
                    f"[*] AI 진단  : 이 정도 {known_type} 수행 능력이면, \n"
                    f"                {target_type}은 [{predicted_target:.1f} kg]을 치는 게 통계적 황금 밸런스입니다.\n"
                    f"[*] 밸런스 갭 : {gap_text}\n"
                    f"--------------------------------------------------\n"
                    f"📢 총평: {advice_comment}\n"
                    f"--------------------------------------------------\n\n"
                    f"💡 실전 바벨 세팅 가이드라인:\n"
                    f" - 오늘 컨디션이 평범하다면 밸런스 목표 중량인 [{predicted_target:.1f}kg]을 잡으세요.\n"
                    f" - 몸이 유난히 가벼운 날(Top Single)에는 최대 [{upper_bound:.1f}kg]까지 증량을 선언해 보십시오.\n"
                    f" - 컨디션이 엉망이고 묵직하다면 안전하게 하한선인 [{lower_bound:.1f}kg]으로 볼륨만 채우고 하차하십시오."
                )
                
                st.write("---")
                # 결과 출력 대시보드 메트릭 카드
                c1, c2, c3 = st.columns(3)
                c1.metric(label="AI 권장 중량", value=f"{predicted_target:.1f} kg")
                c2.metric(label="당일 컨디션 최상 (상한선)", value=f"{upper_bound:.1f} kg")
                
                if balance_gap is not None:
                    delta_val = -balance_gap
                    c3.metric(label="표준 대비 갭 (Gap)", value=f"{balance_gap:.1f} kg", delta=f"{delta_val:.1f} kg")
                    
                st.write("")
                st.text(custom_message)
                
            except Exception as e:
                st.error(f"연산 중 에러가 발생했습니다: {e}")

with tab2:
    st.subheader("🔮 만약에... 내가 몸무게를 더 늘리거나 줄인다면?")
    st.info("사이드바의 체중(kg)과 나이를 조작한 뒤 다시 [스캔 시작]을 눌러보세요. 바뀐 스펙에 맞춰 AI가 미래의 잠재 중량을 실시간으로 재계산합니다.")
