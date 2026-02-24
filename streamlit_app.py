import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime


st.set_page_config(page_title="제주감귤 당도예측기", page_icon="🍊")


@st.cache_resource
def load_model(model_path: str = "brix_model.joblib"):
    return joblib.load(model_path)


def build_features(max_temp: float, min_temp: float, sunshine_hours: float) -> np.ndarray:
    base_input = np.array([[max_temp, min_temp, sunshine_hours]], dtype=float)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    return poly.fit_transform(base_input)


st.title("🍊제주감귤 당도예측기")
st.write("최저기온, 최고기온, 가조시간을 조정하면 브릭스 값을 예측합니다.")

model = load_model()

MIN_TEMP_LOWER = -6.9
MIN_TEMP_UPPER = 28.4
MAX_TEMP_LOWER = -1.4
MAX_TEMP_UPPER = 35.3
MAX_TEMP_SPREAD = 18.8
SUNSHINE_LOWER = 9.9
SUNSHINE_UPPER = 14.4

if "min_temp" not in st.session_state:
    st.session_state.min_temp = 15.0
if "max_temp" not in st.session_state:
    st.session_state.max_temp = 25.0
if "sunshine_hours" not in st.session_state:
    st.session_state.sunshine_hours = 12.0
if "adjust_message" not in st.session_state:
    st.session_state.adjust_message = ""


def normalize_temps(changed_by: str | None = None) -> None:
    min_temp_val = max(MIN_TEMP_LOWER, min(MIN_TEMP_UPPER, float(st.session_state.min_temp)))
    max_temp_val = max(MAX_TEMP_LOWER, min(MAX_TEMP_UPPER, float(st.session_state.max_temp)))

    adjusted_message = ""

    if max_temp_val < min_temp_val:
        if changed_by == "min":
            max_temp_val = min_temp_val
        else:
            min_temp_val = max_temp_val
        adjusted_message = "조건 유지: 최고기온 ≥ 최저기온"

    if max_temp_val - min_temp_val > MAX_TEMP_SPREAD:
        if changed_by == "min":
            max_temp_val = min(MAX_TEMP_UPPER, min_temp_val + MAX_TEMP_SPREAD)
        else:
            min_temp_val = max(MIN_TEMP_LOWER, max_temp_val - MAX_TEMP_SPREAD)
        adjusted_message = "조건 유지: 온도차 ≤ 18.8"

    st.session_state.min_temp = round(min_temp_val, 1)
    st.session_state.max_temp = round(max_temp_val, 1)
    st.session_state.adjust_message = adjusted_message


def on_min_temp_change() -> None:
    normalize_temps(changed_by="min")


def on_max_temp_change() -> None:
    normalize_temps(changed_by="max")


def on_sunshine_change() -> None:
    st.session_state.sunshine_hours = round(
        max(SUNSHINE_LOWER, min(SUNSHINE_UPPER, float(st.session_state.sunshine_hours))),
        1,
    )

normalize_temps()
on_sunshine_change()

left_col, right_col = st.columns(2)

with left_col:
    st.slider(
        "최저기온 (°C)",
        min_value=MIN_TEMP_LOWER,
        max_value=MIN_TEMP_UPPER,
        step=0.1,
        key="min_temp",
        on_change=on_min_temp_change,
    )

    st.slider(
        "최고기온 (°C)",
        min_value=MAX_TEMP_LOWER,
        max_value=MAX_TEMP_UPPER,
        step=0.1,
        key="max_temp",
        on_change=on_max_temp_change,
    )

    st.slider(
        "가조시간 (시간)",
        min_value=SUNSHINE_LOWER,
        max_value=SUNSHINE_UPPER,
        step=0.1,
        key="sunshine_hours",
        on_change=on_sunshine_change,
    )

    if st.session_state.adjust_message:
        st.caption(st.session_state.adjust_message)

min_temp = float(st.session_state.min_temp)
max_temp = float(st.session_state.max_temp)
sunshine_hours = float(st.session_state.sunshine_hours)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_signature" not in st.session_state:
    st.session_state.last_signature = None

try:
    x_pred = build_features(max_temp, min_temp, sunshine_hours)
    pred = model.predict(x_pred)
    result = max(0.0, float(pred[0]))

    with right_col:
        st.success(f"예측 브릭스 값: {result:.2f}")
        st.caption("모델 입력은 다항특성(2차)으로 자동 변환되어 예측됩니다.")
        st.write(f"평균기온(자동 계산): {(min_temp + max_temp) / 2:.1f}°C")
        st.write(f"현재 온도차: {max_temp - min_temp:.1f}°C (최대 {MAX_TEMP_SPREAD})")

    current_signature = (min_temp, max_temp, sunshine_hours)
    if st.session_state.last_signature != current_signature:
        st.session_state.prediction_history.append(
            {
                "시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "최저기온(°C)": round(min_temp, 1),
                "최고기온(°C)": round(max_temp, 1),
                "가조시간(시간)": round(sunshine_hours, 1),
                "예측 브릭스": round(result, 2),
            }
        )
        st.session_state.last_signature = current_signature
except Exception as exc:
    st.error(f"예측 중 오류가 발생했습니다: {exc}")

st.subheader("조회 히스토리")
if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(history_df.iloc[::-1], use_container_width=True)

    st.subheader("입력 변수 대비 당도(브릭스)")
    graph_col1, graph_col2, graph_col3 = st.columns(3)

    with graph_col1:
        st.caption("최저기온 vs 예측 브릭스")
        st.scatter_chart(history_df, x="최저기온(°C)", y="예측 브릭스", use_container_width=True)

    with graph_col2:
        st.caption("최고기온 vs 예측 브릭스")
        st.scatter_chart(history_df, x="최고기온(°C)", y="예측 브릭스", use_container_width=True)

    with graph_col3:
        st.caption("가조시간 vs 예측 브릭스")
        st.scatter_chart(history_df, x="가조시간(시간)", y="예측 브릭스", use_container_width=True)
else:
    st.info("슬라이더 값을 조정하면 여기에 조회 이력이 쌓입니다.")
