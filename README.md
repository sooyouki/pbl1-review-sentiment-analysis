# 📊 계절가전 리뷰 분석 대시보드

대시보드는 두 페이지로 구성되어 있습니다. **실험 비교**는 분류기 개발자가 실험 성능과 오류를 살펴보는 페이지이고, **리뷰 분석 대시보드**는 기업 의사결정자가 제품별 고객 반응과 리뷰 원문을 살펴보는 페이지입니다.

## 1) 페이지 안내

### 실험 비교 — 분류기 개발자용

실험 간 성능 차이와 분류 오류를 분석합니다.

- 실험별 리뷰 수, 감성 분류 수, TP·FP·FN, Precision·Recall·F1
- Aspect_category(대분류) 및 Sub_reason(소분류)별 성능과 F1 비교
- 소분류별 과잉 예측(FP)·누락(FN) 건수와 오답 비율
- 개발자용 리뷰 Drill-down: 평가 단위와 TP·FP·FN을 선택해 모델 예측, Gold 정답, 근거 문구, 원문 비교

### 리뷰 분석 대시보드 — 기업 의사결정자용

제품별 고객 반응에서 주요 불만 신호를 찾고 원문으로 맥락을 확인합니다. 이 페이지에는 TP·FP·FN을 표시하지 않습니다.

- 제품 및 Aspect_category(대분류)별 긍정·부정·중립 언급
- 주요 부정 Sub_reason(소분류), 키워드, 고객 의견
- 부정 비율 30% 이상 50% 미만은 **주의**, 50% 이상은 **경계**로 표시
- 주의·경계 표의 소분류를 선택하면 리뷰 Drill-down이 해당 소분류와 부정 감성으로 필터링
- 리뷰 원문과 모델 근거 문구 확인

## 2) 데이터 준비

`data/` 폴더 안에 공통 파일과 실험별 결과를 구분해 저장합니다. `output/` 아래 실험 폴더는 원하는 만큼 만들 수 있으며, 대시보드가 자동으로 찾아 비교합니다.

```text
data/
├── common/
│   ├── w3_clean.parquet
│   ├── w5_gold_aspects.parquet
│   └── sample_500_ids.csv              # 선택 사항
└── output/
    ├── baseline/
    │   ├── w4_labeled.parquet
    │   ├── w5_metrics_subreason.parquet
    │   └── w5_summary.json
    └── experiment_prompt_v1/
        ├── w4_labeled.parquet
        ├── w5_metrics_subreason.parquet
        └── w5_summary.json
```

### 꼭 필요한 파일

| 파일명 | 위치 | 용도 및 필요한 열 |
|---|---|---|
| `w3_clean.parquet` | `data/common/` | 리뷰 원문과 제품 정보를 연결합니다. `review_id` 또는 `Index`, `RawText_clean` 또는 원문 열, 제품명 열이 필요합니다. |
| `w5_gold_aspects.parquet` | `data/common/` 또는 각 실험 폴더 | 정답 기준입니다. 개발자용 TP·FP·FN Drill-down과 대분류별 성능 계산에 사용합니다. `review_id`, `aspect_category`, `sub_reason`이 필요합니다. |
| `w4_labeled.parquet` 또는 `predictions.parquet` | 각 `data/output/<실험명>/` | 모델의 실험 결과입니다. `review_id`, `aspect_category`, `sub_reason`, `sentiment`, `evidence_text`가 필요합니다. |


### 있으면 좋은 평가 파일

| 파일명 | 위치 | 용도 |
|---|---|---|
| `w5_metrics_subreason.parquet` | 각 실험 폴더 | 소분류별 TP·FP·FN, Precision·Recall·F1을 불러옵니다. 예측 결과와 Gold가 있으면 생략해도 지표를 계산할 수 있습니다. |
| `w5_summary.json` | 각 실험 폴더 | 기존 W5 전체 평가 요약을 표시합니다. |
| `sample_500_ids.csv` | `data/common/` | 분석 표본을 확인할 때 참고합니다. 대시보드 실행에는 필수가 아닙니다. |


## 3) 실행 방법

프로젝트 폴더에서 아래 명령을 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows에서는 가상환경을 다음 명령으로 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

브라우저에서 Streamlit이 안내하는 주소를 엽니다. 왼쪽 **데이터 불러오기**에서 `common/`과 `output/` 폴더가 들어 있는 데이터 루트 `data/`를 선택합니다. 

> ⚠️ 기본 경로는 제 컴퓨터 기준이므로, 사용자 컴퓨터의 경로로 변경해야 합니다. (Streamlit 화면에서도 변경 가능합니다!)

## 4) 사용 순서


### 실험 비교 페이지

1. 페이지 메뉴에서 **실험 비교**를 선택합니다.
2. 전체 지표와 대분류·소분류별 성능 표 및 그래프를 확인합니다.
3. 개발자용 리뷰 Drill-down에서 실험과 평가 단위를 선택합니다.
   - **Sub_reason(소분류):** 세부 분류별 TP·FP·FN 확인
   - **Aspect_category(대분류):** 소분류를 묶은 상위 범주별 TP·FP·FN 확인
4. 제품, 분류, 감성, 평가 결과를 필터링하고 모델 예측과 Gold 정답, 근거 문구, 원문을 비교합니다.

### 리뷰 분석 대시보드 페이지

1. 페이지 메뉴에서 **리뷰 분석 대시보드**를 선택하고 실험을 고릅니다.
2. 제품과 Aspect_category(대분류)를 선택해 감성 분포, 주요 부정 원인과 키워드를 살펴봅니다.
3. 주의·경계 표에서 관심 있는 Sub_reason(소분류)를 선택합니다. 아래 리뷰 Drill-down이 해당 소분류와 부정 감성으로 자동 필터링됩니다.
4. 실제 리뷰 원문과 근거 문구를 확인합니다. 필요하면 Drill-down에서 필터를 바꾸거나 페이지를 넘기며 여러 리뷰를 살펴봅니다.
