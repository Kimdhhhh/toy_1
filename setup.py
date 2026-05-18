import pandas as pd
import numpy as np
import logging
import time
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, MaxAbsScaler, Normalizer,
    OneHotEncoder, OrdinalEncoder
)

strategy_map = {
    # 1. 모든 수치형 변수(타겟 변수 제외)를 S5로 통일
    'S5': [
            'Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'KgWeightClass', 'Year', 'Month', 'Day', 'Disq'
    ],
    
    # 2. 모든 범주형 변수를 E1으로 처리
    'E1': [
        'Sex', 'Event', 'Equipment'
    ],
}

# 판다스 데이터프레임으로 결과 반환 고정
set_config(transform_output="pandas")

# 로거 설정
logger = logging.getLogger("Preprocessor")
if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# [Scaling Strategy Library]
SCALING_STRATEGIES = {
    "S1": Pipeline([('power', PowerTransformer()), ('scaler', StandardScaler())]),
    "S2": Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler())]),
    "S3": Pipeline([('maxabs', MaxAbsScaler()), ('norm', Normalizer())]),
    "S4": Pipeline([('scaler', StandardScaler()), ('norm', Normalizer(norm='l2'))]),
    "S5": Pipeline([('scaler', StandardScaler())])
}

def get_encoding_strategy(strategy_id, category_orders=None, features=None):
    """[Encoding Strategy Library]"""
    if strategy_id == 'E1':
        return Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    elif strategy_id == 'E2':
        return Pipeline([('label', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    elif strategy_id == 'E3':
        if category_orders and features:
            orders = [category_orders.get(col, []) for col in features]
            return Pipeline([('ordinal', OrdinalEncoder(categories=orders, handle_unknown='use_encoded_value', unknown_value=-1))])
        else:
            return Pipeline([('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    return None

def apply_preprocessing(X_input, strategy_map, category_orders=None, is_train=True, preprocessor=None):
    """
    전략 기반 통합 전처리 파이프라인 (X 피처 전용)
    """
    start_total = time.time()
    mode = "TRAIN" if is_train else "TEST/INFERENCE"
    logger.info(f"--- START PREPROCESSING [{mode} MODE] ---")
    
    try:
        if is_train:
            transformers = []
            for s_id, features in strategy_map.items():
                if s_id.startswith('S'):
                    pipe = SCALING_STRATEGIES.get(s_id, Pipeline([('scaler', StandardScaler())]))
                elif s_id.startswith('E'):
                    pipe = get_encoding_strategy(s_id, category_orders, features)
                else:
                    continue
                transformers.append((f"pipe_{s_id}", pipe, features))
            
            preprocessor = ColumnTransformer(transformers, remainder='passthrough')
            df_processed = preprocessor.fit_transform(X_input)
            
        else:
            if preprocessor is None:
                raise ValueError("Test 모드(is_train=False)에서는 학습된 preprocessor 객체를 반드시 전달해야 합니다.")
            df_processed = preprocessor.transform(X_input)

    except Exception as e:
        logger.error(f"Preprocessing Failed: {str(e)}")
        raise
    
    # 💡 [핵심 수정 부분] Numpy 배열이든 Pandas 표든 안전하게 처리하는 방어 로직
    if isinstance(df_processed, pd.DataFrame):
        df_processed.index = X_input.index
    else:
        try:
            cols = preprocessor.get_feature_names_out()
        except:
            cols = None
        df_processed = pd.DataFrame(df_processed, index=X_input.index, columns=cols)
        
    logger.info(f"--- FINISH [{mode} MODE] (Time: {time.time() - start_total:.4f}s) ---")
    
    return df_processed, preprocessor