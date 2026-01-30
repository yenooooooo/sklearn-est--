# Scikit-Learn Machine Learning Study with Google Antigravity

이 저장소는 **Google Antigravity** 에이전트와 함께 진행한 머신러닝 학습 및 실습 기록입니다.  
Scikit-learn을 중심으로 데이터 전처리, 모델 학습, 평가, 그리고 고급 튜닝(Optuna, AutoML)까지 다양한 주제를 다루며, AI 페어 프로그래밍을 통해 작성되었습니다.

## 📂 프로젝트 구성

이 프로젝트는 Antigravity와의 대화를 통해 생성되고 발전된 코드들로 구성되어 있습니다.

### 1. Scikit-Learn 기초 및 핵심 알고리즘
- **기초 및 전처리**: `1_sklearn_start.ipynb`, `4_sklearn_PreProcess.ipynb`
- **모델 선택 및 평가**: `2_ModelSelection.ipynb`
- **지도 학습 알고리즘**:
  - `3_SVM.ipynb` (Support Vector Machine)
  - `5_sklearn_classification.ipynb` (분류 모델 종합)
  - `9_LinearRegressionModel.ipynb` (선형 회귀)
  - `10_ensemble.ipynb` (앙상블 기법)
- **최적화**: `6_classification_Optuna.ipynb`, `11_ensemble_Optuna.ipynb` (Optuna를 활용한 튜닝)
- **비지도 학습**: `13_unsupervisedLearning.ipynb`

### 2. 실전 프로젝트 및 데이터 분석 (EDA)
실제 데이터셋을 활용한 심화 분석 및 예측 프로젝트입니다.
- **Titanic Survival Prediction**: `7_Titanic.ipynb`, `타이타닉생존자예측-*.ipynb`
  - 데이터 전처리, 모델링, 캐글(Kaggle) 제출 실습
- **Wine Quality Analysis**: `Plus_1_sklearn_wine_classification.ipynb`, `Plus_2_Red_wine_quality_analysis.ipynb`
- **Digits Classification**: `Plus_2_sklearn_digits.ipynb`, `Plus_3_sklearn_digits.ipynb`
- **California Housing**: `GBM_visualize.ipynb` (Gradient Boosting 시각화)

### 3. AutoML 및 기타
- **AutoML**: `AutoGluon`을 활용한 자동화된 머신러닝 모델 학습 (`AutoML/`, `AutogluonModels/`)
- **Web ML Integration**: 머신러닝 모델의 웹 서비스 연동 실습 (`webML/`)
- **개인 공부**: 추가적인 개인 학습 자료 (`개인공부/`)

## 🛠 사용 기술 (Tech Stack)
- **Environment**: **Google Antigravity** (AI Coding Assistant)
- **Language**: Python
- **Libraries**:
  - `scikit-learn`: 머신러닝 핵심 라이브러리
  - `pandas`, `numpy`: 데이터 처리
  - `matplotlib`, `seaborn`: 데이터 시각화
  - `optuna`: 하이퍼파라미터 최적화
  - `autogluon`: AutoML 프레임워크

## 🚀 사용 방법 (How to Use)
이 저장소의 코드는 **Google Antigravity**와의 상호작용을 통해 작성되고 실행되었습니다.
- 각 `.ipynb` 파일은 학습 과정과 코드 실행 결과를 담고 있습니다.
- Antigravity 에이전트를 통해 특정 개념에 대해 질문하거나, 코드의 수정 및 디버깅을 요청하며 학습을 진행했습니다.

