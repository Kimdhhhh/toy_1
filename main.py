import sys
import os

# 💡 [보정 1] 자식 프로세스 경로 유실 방지 치트키 (무조건 1번 라인 고정)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field
from supabase import create_client, Client
from python.config import SUPABASE_URL, SUPABASE_KEY
from typing import Optional
import pandas as pd
import joblib
import datetime
import pprint
from setup import apply_preprocessing, strategy_map

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="SBD AI 스트렝스 밸런스 진단 서비스",
    description="""
    108만 명의 빅데이터 기반 AI 모델을 통해 유저의 종목별 황금 비율을 계산하고,
    신체 불균형 진단 및 당일 컨디션별 맞춤형 중량 가이드라인을 제공합니다.
    """,
    version='2.0.0',
)

try:
    artifact = joblib.load('SBD_v2.pkl') 
    model = artifact['model']
    saved_preprocessor = artifact['preprocessor']
    mae_value = artifact['mae']
    print(f"✅ [SUCCESS] SBD AI 준비물 로드 성공! (확인된 MAE 오차값: {mae_value})")
except Exception as e:
    print(f"🚨 [CRITICAL] 아티팩트 로드 실패. 파일명을 확인하세요: {e}")
    
# =====================================================================
# 2. Pydantic 입출력 데이터 스키마 정의
# =====================================================================
class SBDRequest(BaseModel):
    Sex: str = Field(..., description="성별 (M: 남성, F: 여성)")
    Age: float = Field(..., description="사용자 나이 (만 나이)", ge=17, le=100)
    BodyweightKg: float = Field(..., description="현재 체중 (kg)", ge=30, le=300)
    Equipment: str = Field(..., description="사용 장비 (Raw, Wraps, Single-ply 등)")
    Known_Type: str = Field(..., description="현재 기준 삼을 자신 있는 종목 (Squat, Bench, Deadlift)")
    Known_1RM: float = Field(..., description="기준 종목의 현재 1RM 기록 (kg)", ge=0, le=600)
    Target_Type: str = Field(..., description="AI가 예측/진단해 줄 타겟 종목 (Squat, Bench, Deadlift)")
    Target_Actual_1RM: Optional[float] = Field(None, description="유저가 알고 있는 타겟 종목의 실제 기록 (선택 입력)", ge=0, le=600)

    @field_validator('Age')
    @classmethod
    def age_check(cls, v):
        if not (17 <= v <= 100): raise ValueError('나이는 17세에서 100세 사이여야 합니다.')
        return v

    @field_validator('BodyweightKg')
    @classmethod
    def weight_check(cls, v):
        if not (30 <= v <= 300): raise ValueError('체중이 상식 범위에서 벗어납니다.(30~300Kg)')
        return v

    model_config = {
        'json_schema_extra': {
            "example": {
                "Sex": "M", "Age": 30.0, "BodyweightKg": 101.0, "Equipment": "Raw",
                "Known_Type": "Squat", "Known_1RM": 190.0, "Target_Type": "Bench",
                "Target_Actual_1RM": 130.0  
            }
        }
    }
    
class PredictionResponse(BaseModel):
    status: str
    predicted_target_1rm: float
    confidence_interval: dict
    balance_gap_kg: Optional[float]
    message: str

# =====================================================================
# 3. MLOps 비동기 Supabase 로깅 함수 및 예외 핸들러
# =====================================================================
def log_to_supabase_v2(status: str, db_snapshot: dict, error_msg: str = None):
    log_entry = {
        "status": status,
        "requested_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_sex": db_snapshot.get("Sex"),
        "user_age": db_snapshot.get("Age"),
        "user_bodyweight_kg": db_snapshot.get("BodyweightKg"),
        "user_equipment": db_snapshot.get("Equipment"),
        "known_type": db_snapshot.get("Known_Type"),
        "known_1rm": db_snapshot.get("Known_1RM"),
        "target_type": db_snapshot.get("Target_Type"),
        
        "derived_age_from_peak": round(abs(db_snapshot.get("Age", 32) - 32.0), 4),
        "derived_rel_strength": round(db_snapshot.get("Known_1RM", 0) / db_snapshot.get("BodyweightKg", 1), 4),
        "predicted_target_1rm": db_snapshot.get("predicted_target_1rm"),
        "mae_applied": db_snapshot.get("mae_applied"),
        "user_reported_actual_1rm": db_snapshot.get("Target_Actual_1RM"),
        "balance_gap_kg": db_snapshot.get("balance_gap_kg"),
        "error_msg": error_msg
    }
    try:
        sb.table("inference_logs").insert(log_entry).execute() 
    except Exception as e:
        print(f'[!] 수파베이스 로그 전송 실패 : {e}')

# 💡 [보정 2] 유실되었던 데이터 검증 에러 핸들러 복구
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    error_details = exc.errors()
    log_entry = {
        "status": "VALIDATION_ERROR",
        "predicted_target_1rm": 0,
        "mae_applied": 0,
        "request_json": {"body": exc.body if hasattr(exc, 'body') else "No Body"},
        "error_msg": str(error_details) 
    }
    try:
        sb.table("inference_logs").insert(log_entry).execute()
    except Exception as e:
        print(f"[!] 에러 로깅 실패: {e}")

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "입력 데이터 형식이 올바르지 않습니다.",
            "details": error_details
        },
    )

# =====================================================================
# 4. 실시간 추론 핵심 엔드포인트
# =====================================================================
@app.post('/predict', response_model=PredictionResponse)
async def predict_total_kg(data: SBDRequest, background_tasks: BackgroundTasks):
    input_dict = data.model_dump()
    
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
        
        if data.Target_Actual_1RM is not None:
            balance_gap = predicted_target - data.Target_Actual_1RM
            gap_status = "부족" if balance_gap > 0 else "초과 우수"
            gap_text = f"표준 비율 대비 현재 약 {abs(balance_gap):.1f} kg {gap_status} 상태입니다."

        # [E] 동적 팩폭 문구 빌드
        custom_message = (
            f"--------------------------------------------------\n"
            f"🏋️‍♂️ AI SBD 스트렝스 밸런스 진단 리포트\n"
            f"--------------------------------------------------\n"
            f"[*] 현재 스펙 : {data.Sex} / {data.Age:.0f}세 / {data.BodyweightKg:.1f}kg / {data.Equipment}\n"
            f"[*] 기준 입력 : {data.Known_Type} {data.Known_1RM:.1f} kg\n"
            f"[*] AI 진단  : 이 정도 스쿼트면, {data.Target_Type}은 원래 {predicted_target:.1f} kg 쳐야 정석 비율입니다.\n"
            f"[*] 현재 상태 : {gap_text}\n"
            f"--------------------------------------------------\n\n"
            f"💡 AI 실전 컨디션 가이드라인:\n"
            f" - 오늘 컨디션이 평범하다면 정석 밸런스 무게인 [{predicted_target:.1f}kg]을 타겟으로 잡으세요.\n"
            f" - 몸이 가벼운 날(Top Single)에는 최대 [{upper_bound:.1f}kg]까지 증량을 시도하셔도 통계적으로 안전합니다.\n"
            f" - 컨디션이 저조하다면 안전하게 하한선인 [{lower_bound:.1f}kg]으로 볼륨을 채우십시오."
        )

        # [F] MLOps 비동기 로깅 스냅샷 패킹 및 태스크 추가
        log_snapshot = {**input_dict, "predicted_target_1rm": predicted_target, "mae_applied": MAE_VALUE, "balance_gap_kg": balance_gap}
        
        background_tasks.add_task(
            log_to_supabase_v2,
            status="SUCCESS",
            db_snapshot=log_snapshot
        )

        # [G] 즉시 응답 반환
        return {
            "status": "success",
            "predicted_target_1rm": round(predicted_target, 1),
            "confidence_interval": {
                "lower_kg": round(lower_bound, 1),
                "upper_kg": round(upper_bound, 1)
            },
            "balance_gap_kg": round(balance_gap, 1) if balance_gap is not None else None,
            "message": custom_message  
        }

    except Exception as e:
        background_tasks.add_task(
            log_to_supabase_v2,
            status="FAIL",
            db_snapshot=input_dict,
            error_msg=str(e)
        )
        raise HTTPException(status_code=500, detail="서버 내부 연산 중 에러가 발생했습니다.")