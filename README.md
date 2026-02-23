# 🍊 Orange Brix Predictor App

코랩에서 학습한 `brix_model.joblib` 모델을 사용해 브릭스(Brix) 값을 예측하는 Streamlit 앱입니다.

## 입력 변수

- 최고기온 (°C)
- 최저기온 (°C)
- 일조시간 (시간)

앱 내부에서 위 3개 입력값을 2차 다항특성으로 변환한 뒤 모델에 전달합니다.

## 실행 방법

1. 의존성 설치

   ```bash
   pip install -r requirements.txt
   ```

2. 앱 실행

   ```bash
   streamlit run streamlit_app.py
   ```
