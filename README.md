# 계절가전 리뷰 분석 대시보드

Streamlit 대시보드로 리뷰 분류 실험을 비교하고, 예측 결과와 고객 리뷰 원문을 확인합니다. 화면은 **분류기 개발자용 실험 비교**와 **기업 의사결정자용 리뷰 분석** 두 페이지로 구성되어 있습니다.

## 교수님 대시보드 필수 요구사항 확인

강의안의 5가지 필수 기능을 현재 구현과 대조했습니다. 다섯 항목 모두 대시보드에서 확인할 수 있습니다.

| 강의안 요구사항 | 구현 위치 및 확인 방법 | 상태 |
|---|---|---|
| 1. 제품 또는 카테고리 선택 | 리뷰 분석 대시보드 상단에서 제품과 Aspect_category(대분류)를 선택합니다. 선택 범위가 차트, 키워드, 부정 비율 표, 리뷰 Drill-down에 적용됩니다. | 충족 |
| 2. Aspect별 긍정·부정 현황 | Aspect_category(대분류)별 긍정·부정·중립 언급 수를 막대그래프로 보여줍니다. | 충족 |
| 3. Sub-reason별 주요 부정 원인 | 부정 언급이 많은 Sub_reason(소분류)를 건수 그래프로 보여줍니다. | 충족 |
| 4. 주요 Keyword 또는 주요 의견 | 부정 리뷰에서 주요 키워드를 집계하고, 리뷰 근거 문구 또는 원문에 포함된 고객 의견을 함께 보여줍니다. | 충족 |
| 5. 특정 결과 선택 후 근거 원문 확인 | 주의·경계 소분류 표에서 행을 선택하면 리뷰 Drill-down이 해당 소분류와 부정 감성으로 자동 필터링됩니다. 아래에서 실제 리뷰 원문과 모델 근거 문구를 확인할 수 있습니다. | 충족 |

요구사항 5는 **부정 비율 표에서 행을 클릭해 관련 리뷰로 이동하는 흐름**으로 구현했습니다. 리뷰 Drill-down에서도 소분류와 감성을 직접 선택할 수 있습니다.

## 페이지 구성

### 실험 비교 · 분류기 개발자용

- 실험별 Total 리뷰 수, 감성 분류 수, TP·FP·FN, 전체 Precision·Recall·F1 비교
- Aspect_category(대분류)와 Sub_reason(소분류)별 Precision·Recall·F1 비교
- 대분류·소분류별 F1 그래프
- 소분류별 과잉(FP)·누락(FN) 건수와 오답 내 비율 비교
- 개발자용 리뷰 Drill-down: 평가 단위를 대분류 또는 소분류로 선택하고 TP·FP·FN별 모델 예측, Gold 정답, 근거 문구와 원문 확인

### 리뷰 분석 대시보드 · 기업 의사결정자용

- 실험, 제품, Aspect_category(대분류) 선택
- 대분류별 감성 분포, 주요 부정 소분류, 키워드와 고객 의견 확인
- 부정 비율 **30% 이상 50% 미만은 주의**, **50% 이상은 경계**로 표시
- 주의·경계 표의 소분류를 선택하면 리뷰 Drill-down이 해당 소분류와 부정 감성으로 자동 필터링
- 고객용 Drill-down에는 TP·FP·FN을 표시하지 않고, 리뷰 원문과 분류된 대분류·소분류·감성·근거 문구를 표시

## 실행 방법

Python 환경에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows에서는 가상환경 활성화 명령을 다음처럼 사용합니다.

```powershell
.venv\Scripts\Activate.ps1
```

브라우저에서 Streamlit이 출력하는 주소를 엽니다. 데이터 폴더를 기본 경로에서 찾지 못하면 왼쪽 **데이터 불러오기** 영역에서 본인 컴퓨터의 데이터 루트 폴더를 입력합니다.

## 데이터 폴더 구조

폴더 불러오기에서 아래와 같은 구조를 사용합니다. 실험 폴더 이름과 개수는 자유롭게 정할 수 있으며, `data/output/` 바로 아래의 실험별 폴더를 자동으로 찾아 비교합니다.

```text
data/
├── common/
│   ├── w3_clean.parquet
│   ├── sample_500_ids.csv          # 선택 사항: 표본 추적용
│   └── w5_gold_aspects.parquet     # 실험 공통 Gold
└── output/
    ├── baseline/
    │   ├── predictions.parquet
    │   ├── w5_metrics_subreason.parquet
    │   └── w5_summary.json
    ├── precision_recall_prompt/
    │   ├── predictions.parquet
    │   ├── w5_metrics_subreason.parquet
    │   └── w5_summary.json
    └── safety_tag_prompt/
        ├── predictions.parquet
        ├── w5_metrics_subreason.parquet
        └── w5_summary.json
```

같은 Gold 데이터를 쓰는 실험이면 Gold 파일은 `common/`에 한 번만 두면 됩니다. 특정 실험에서 다른 Gold를 사용한 경우에는 해당 실험 폴더 안에 `w5_gold_aspects.parquet`을 넣으면 공통 Gold보다 우선해서 사용합니다.

### 파일과 필수 열

| 파일 | 용도 | 주요 열 |
|---|---|---|
| `w3_clean.parquet` | 리뷰 원문, 제품 정보 연결 | `review_id`, `RawText_clean`, 제품명 열 |
| `w5_gold_aspects.parquet` | 정답 및 TP·FP·FN 평가 | `review_id`, `sub_reason`; 대분류 지표에는 `aspect_category` 필요 |
| `predictions.parquet` | 실험별 모델 결과 | `review_id`, `aspect_category`, `sub_reason`, `sentiment`, `evidence_text` |
| `w5_metrics_subreason.parquet` | 소분류별 평가 지표 | `sub_reason`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1` |
| `w5_summary.json` | 전체 실험 지표 요약 | `sub_reason`, `aspect_category` 지표 객체 등 기존 W5 요약 형식 |

예측 파일은 `predictions.parquet`을 권장합니다. 기존 `w4_labeled.parquet` 이름도 인식합니다. 실제 파일의 열 이름이 다르더라도 앱에서 일부 표준 별칭(`Index`, `Aspect`, `SentimentPolarity`, `SentimentText` 등)을 자동으로 인식합니다.

## 계산 및 해석 기준

- Aspect 추출 평가는 `(review_id, 분류 라벨)` 쌍을 기준으로 중복을 제거합니다. 예측과 정답이 일치하면 TP, 예측에만 있으면 FP, 정답에만 있으면 FN입니다.
- 부재 클래스 TN은 이 추출 평가에서 계산하지 않습니다.
- 전체 감성 분류 수는 중복 `(review_id, sub_reason)` 쌍을 제거한 뒤 긍정·부정·중립 감성이 있는 예측 수입니다.
- 주의·경계 부정 비율은 현재 선택한 실험과 제품·대분류 범위 안에서, 해당 소분류가 예측된 리뷰 중 부정으로 분류된 비율입니다. 이 값은 제품 전체의 확정적 위험도나 인과관계를 뜻하지 않으므로 원문과 표본 수를 함께 확인합니다.
- 소분류별 성능 표는 `w5_metrics_subreason.parquet`을 사용합니다. 대분류별 성능은 예측 결과와 Gold 파일에 `aspect_category`가 있을 때 계산됩니다.

## 팀 공유 체크리스트

팀원에게는 이 폴더의 `app.py`, `requirements.txt`, `README.md`와 위 구조의 데이터 폴더를 함께 공유하면 됩니다. 데이터에는 실제 리뷰 원문이 포함될 수 있으므로, 팀에서 접근 가능한 공유 위치에만 저장하세요.
