# [중요] AFB 등급 점수화 — 2단 구조 (EER + AFB 축 calibration)

> **목적:** D1 AFB `1+`~`4+`에 **임의 등간격 점수**(`afb_grade_v1`) 대신,  
> **통계적 근거가 있는 2단 체계**로 infectivity 라벨·스코어를 정의한다.  
>
> **상태:** 설계 문서 (구현·실험은 코호트·영상 DB 확보 후)  
> **관련:** Phase III EER (`TB Phase III/artifacts/eer_relabel_260621/`), v1 graded labels (`src/labels/`), **CXR MTL 스펙** [`docs/CXR_MULTITASK_SPEC.md`](docs/CXR_MULTITASK_SPEC.md) (D1 primary = polychoric)  
> **감사 리포트:** `artifacts/label_audit_report.md` (JSON 전수 스캔)

---

## 0. AFB 데이터 및 현재 출력값 (코호트 현황)

### 0.1 데이터 소스·파싱

| 항목 | 내용 |
|------|------|
| **원본** | `D:\[FINAL] SPUTUM DATA\*.json` — `test_result_blocks[]` |
| **추출** | 결과문 `AFB stain: …` 줄 (`AFB_STAIN_RE` in `src/labels/utils.py`) |
| **검체** | 객담 (`sputum` / `객담` in `test_name`) |
| **날짜** | 신고일자 anchor ± **2개월** (`window_months: 2`) 안 검사만 |
| **study 집계** | 윈도우 내 AFB 줄마다 점수화 후 **`max(afb_values)`** → D1 1값 (`d1_note`: `afb_max=…`) |
| **코호트** | 강남 1xxxx only; registry union **1601** study (`labels_v1.csv`) |

동일 study에 AFB 검사가 여러 번 있으면 **가장 높은 등급**이 D1이 됩니다 (랜덤 선택 아님).

### 0.2 JSON 원문 — AFB stain 줄 (8종)

`audit_labels.py` 전수 스캔 기준 (**검사 줄** 단위, n=**14,118** 줄):

| 건수 | 원문 (raw text) | 비고 |
|-----:|-----------------|------|
| 11,263 | `No AFB seen` | 음성 |
| 1,338 | `Rare (1+)` | |
| 539 | `Few (2+)` | |
| 417 | `Moderate (3+)` | |
| 387 | `Doubtful Repeat test (Trace)` | trace |
| 112 | `Numerous (4+)` | |
| 56 | `Negative for AFB` | 음성 (동의어) |
| 6 | `Positive for AFB` | 등급 미기재 양성 → **1.0** |

**고유 문자열 8개.** `1+` / `2+` / `3+` / `4+` 외에 trace·무분류 양성이 존재.

### 0.3 현재 baseline — `afb_grade_v1` (D1 출력)

구현: `src/labels/encoders.py` → `encode_afb_grade_v1`

| 파서 입력 (원문 패턴) | D1 점수 | severity |
|----------------------|--------:|----------|
| `No AFB seen`, `Negative for AFB`, `not seen`, `negative` | **0.0** | 음성 |
| `Doubtful`, `Trace` (예: `Doubtful Repeat test (Trace)`) | **0.125** | trace |
| `Rare (1+)`, `1+` | **0.25** | 1+ |
| `Few (2+)`, `2+` | **0.50** | 2+ |
| `Moderate (3+)`, `3+` | **0.75** | 3+ |
| `Numerous (4+)`, `4+` | **1.0** | 4+ |
| `Positive for AFB` (등급 없음) | **1.0** | 양성 상한 |
| 미매칭 / 윈도우 밖 / 비객담 | *(missing)* | D1 빈칸 |

### 0.4 study-level `labels_v1.csv` — D1 분포

윈도우·집계(`max`) 후 **study당 스칼라 1개** (D1 labeled **n=1,377**):

| D1 (`afb_grade_v1`) | study 수 | 비율 |
|--------------------:|---------:|-----:|
| 0.0 | 749 | 54.4% |
| 0.125 (trace) | 58 | 4.2% |
| 0.25 (1+) | 237 | 17.2% |
| 0.50 (2+) | 121 | 8.8% |
| 0.75 (3+) | 154 | 11.2% |
| 1.0 (4+ / Positive for AFB) | 58 | 4.2% |

- **음성 0**이 과반 → full-cohort PLS 시 음성/양성 축에 끌림 (§8 참고).
- **4+ / 1.0** study **58명** → 2층 \(f(\text{grade})\) 추정 시 tail 불안정.
- **고유 출력값 6개** (0.0, 0.125, 0.25, 0.5, 0.75, 1.0).

### 0.5 등록된 AFB 인코더 (ablation용)

`src/labels/encoders.py` — `build_graded_labels.py --afb-scheme` 로 선택:

| scheme | 설명 | study-level 점수 분포 (요약) |
|--------|------|------------------------------|
| **`afb_grade_v1`** | 임상 등간격 (현재 default) | 0 / 0.125 / 0.25 / 0.5 / 0.75 / 1.0 |
| `afb_binary` | 양성>0 → 1 | 0.0×11319줄, 1.0×2799줄 (줄 기준 audit) |
| `afb_ordinal_5` | trace→0.25, 나머지 등급 유지 | 0 / 0.25 / 0.5 / 0.75 / 1.0 |
| `afb_raw_div4` | `N+` → N/4 (trace→0.25) | 0 / 0.25 / 0.5 / 0.75 / 1.0 |

### 0.6 2층 calibration용 ordinal 라벨 (개념)

EER·PLS·ridit 등 **2층**에서 쓸 **순서형 등급** (음성 제외 시):

| ordinal | 원문 그룹 | `afb_grade_v1` |
|--------:|-----------|---------------:|
| 0 | 음성 | 0.0 |
| 1 | trace | 0.125 |
| 2 | 1+ | 0.25 |
| 3 | 2+ | 0.50 |
| 4 | 3+ | 0.75 |
| 5 | 4+ / Positive for AFB | 1.0 |

---

## 1. 문제 정의

### 1.1 현재 `afb_grade_v1`의 한계

§0.3 점수표는 **임상 등간격 가정** — 통계적 검증 없음.  
2층에서 \(f(\text{grade})\)를 데이터로 다시 잡는 것이 본 문서의 목표.

### 1.2 원하는 것

| 질문 | 원하는 답 |
|------|-----------|
| AFB **채널**이 전체 infectivity에서 얼마나 중요한가? | **데이터 기반 가중치** (EER) |
| AFB **채널 안에서** 1+ vs 4+ 간 거리는? | **단조 함수** \(f(\text{grade}) \in [0,1]\) |
| “이 환자 감염력 N점” (gold standard) | **존재하지 않음** — 척도 **정의(calibration)** 문제 |

---

## 2. 핵심: 2단 구조

```
┌─────────────────────────────────────────────────────────────┐
│  [1층] EER — 패널 레벨 (D1~D5 head 간 가중치)                  │
│  w_D5, w_D1(AFB), w_D2(PCR), w_D3(solid), w_D4(liquid)      │
│  → "AFB 채널이 전체에서 차지하는 몫"                          │
└─────────────────────────────────────────────────────────────┘
                              ×
┌─────────────────────────────────────────────────────────────┐
│  [2층] AFB 축 only — 등급별 점수 f(grade) ∈ [0,1]            │
│  단조: 0 < trace < 1+ < 2+ < 3+ < 4+  (또는 양성 subset만)    │
│  → "그 채널 안에서 1+와 4+ 간 거리"                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 합성 infectivity score (개념)

Phase III EER 프레임과 동일. D1만 **연속화**:

\[
S = w_5 \cdot D5 + w_1 \cdot f(\text{AFB grade}) + w_2 \cdot D2 + w_3 \cdot D3 + w_4 \cdot D4
\]

- \(w_k\): **1층 EER**에서 산출 (study-level, D1~D5 polychoric + effective rank + micro PLS).
- \(f(\cdot)\): **2층**에서 산출 (등급 → [0,1], **단조**).
- \(D2\)~\(D5\): v1 연속 라벨 또는 Phase III 이진 — 스코어 정의 시 버전 고정.

**역할 분리**

| 층 | 담당 | 하지 않는 것 |
|----|------|--------------|
| 1층 EER | head 간 **절대 비중** | 1+ vs 4+ **등급 간 거리** |
| 2층 \(f\) | AFB **등급 간 shape** | 다른 head(D2~D5) 비중 |

---

## 3. [1층] EER — D1~D5 패널 가중치

### 3.1 방법 (Phase III 재사용)

1. study-level complete-case 테이블 (D1~D5).
2. **Polychoric** 상관 행렬.
3. **Entropy effective rank**: Imaging (D5 Cavity) vs Microbiology (D1~D4) 도메인 분할.
4. Micro 블록 내 **PLS Mode-A outer weights** → D1, D2, D3, D4 상대 비중.
5. 최종 \(w_1 \ldots w_5\) (합 = 1).

**참고 산출물**

- `D:\TB Phase III\artifacts\eer_relabel_260621\eer_relabel_recompute.json`
- 예시 (relabel, n=687): D5≈0.36, D1≈0.15, D2≈0.15, D3≈0.17, D4≈0.16

### 3.2 EER만으로 AFB 등급별(1+, 4+) 비중을 주지 못하는 이유

- EER 입력은 **head 단위** (D1 하나의 이진/순서형 열).
- 1+, 2+, 3+, 4+를 **각각 이진 지표**로 넣으면 **nested** (4+ ⊂ 3+ ⊂ …) → 독립 가정 깨짐.
- 객담 지표를 micro 블록에 **무작정 많이** 넣으면 effective rank·PLS가 **중복 축**을 나눠 가져 비중이 왜곡됨.

**→ EER 1층은 D1~D5 다섯 head만 유지. AFB 세부 등급은 2층으로 분리.**

---

## 4. [2층] AFB 축 — \(f(\text{grade})\) 산출 방법

공통 요구사항:

- 출력: 등급별 \(f \in [0,1]\).
- **단조성**: severity 순서 유지 (1+ < 2+ < … < 4+; 0은 최하).
- 논문 supplement에 **점수표 고정** 후 ViT·RF 학습에 사용.

### 4.1 방법 비교 요약

| 방법 | 입력 | AFB 축 only | 단조 | 에비던스 성격 | 비고 |
|------|------|-------------|------|---------------|------|
| **afb_grade_v1** | 임상 규칙 | ✅ | ✅ (설계상) | 임상 직관 | baseline |
| **Ridit** | 코호트 등급 빈도 | ✅ | ✅ (순서형) | **역학 분포** 위치 | 영상 불필요, 구현 쉬움 |
| **PLS (AFB+ subset)** | CXR feature + ordinal Y | ✅ (양성만) | 검증 후 보정 | **영상–도말 정합** | n 작음, CV 필수 |
| **Polychoric ordinal** | AFB ordinal ↔ D3/D4 | ✅ | 상관 구조 | **객담 내부 정합** | 영상 불필요 |
| **Optimal scaling** | AFB + 앵커 변수 | ✅ | 최적화에 따라 | **다지표 정합** | 앵커 선택 중요 |
| **Isotonic + anchor** | anchor 연속값 | ✅ | **강제** | anchor와 단조 맞춤 | culture TTP 등 |

### 4.1.1 실측 병렬 비교표 (labels_v1, n=1377)

> 산출: `scripts/run_afb_layer2_comparison.py` → [`artifacts/afb_layer2_comparison.json`](artifacts/afb_layer2_comparison.json)  
> 코호트: study-level `labels_v1.csv` (D1 labeled n=1377). Culture 앵커 = z(D3)+z(D4). PLS = v7.011 DenseNet CXR feature, AFB+ subset (n=314).

| Raw text | ord | grade_v1 | ridit | pls_cxr | polychoric | optimal | isotonic |
|----------|----:|---------:|------:|--------:|-----------:|--------:|---------:|
| `No AFB seen` | 0 | 0.000 | 0.272 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Negative for AFB` | 0 | 0.000 | 0.272 | 0.000 | 0.000 | 0.000 | 0.000 |
| `Doubtful Repeat test (Trace)` | 1 | 0.125 | 0.565 | 1.000 | 0.474 | 0.474 | 0.849 |
| `Rare (1+)` | 2 | 0.250 | 0.672 | 1.000 | 0.603 | 0.603 | 0.885 |
| `Few (2+)` | 3 | 0.500 | 0.802 | 1.000 | 0.686 | 0.686 | 0.912 |
| `Moderate (3+)` | 4 | 0.750 | 0.902 | 1.000 | 0.940 | 0.940 | 0.953 |
| `Numerous (4+)` | 5 | 1.000 | 0.979 | 1.000 | 1.000 | 1.000 | 1.000 |
| `Positive for AFB` | 5 | 1.000 | 0.979 | 1.000 | 1.000 | 1.000 | 1.000 |

**등급별 f(g)**

| grade | ord | grade_v1 | ridit | polychoric | optimal | isotonic | pls_cxr |
|-------|----:|---------:|------:|-----------:|--------:|---------:|--------:|
| 음성 | 0 | 0.000 | 0.272 | 0.000 | 0.000 | 0.000 | 0.000 |
| trace | 1 | 0.125 | 0.565 | 0.474 | 0.474 | 0.849 | 1.000 |
| 1+ | 2 | 0.250 | 0.672 | 0.603 | 0.603 | 0.885 | 1.000 |
| 2+ | 3 | 0.500 | 0.802 | 0.686 | 0.686 | 0.912 | 1.000 |
| 3+ | 4 | 0.750 | 0.902 | 0.940 | 0.940 | 0.953 | 1.000 |
| 4+ | 5 | 1.000 | 0.979 | 1.000 | 1.000 | 1.000 | 1.000 |

**해석 메모**

| 방법 | n_used | 단조 | 핵심 수치 |
|------|-------:|------|-----------|
| ridit | 1377 | OK | 음성 ridit≈0.27 (코호트 54% 음성 → 누적 중앙) |
| polychoric | 1377 | OK | ρ(AFB,D3)≈0.79, ρ(AFB,D4)≈0.73 |
| optimal_scaling | 1377 | OK | corr(anchor)≈0.64 |
| isotonic_anchor | 1377 | OK | trace~1+ median anchor 거의 동일 → 상위 등급 압축 |
| pls_cxr | 314 | OK* | R²≈0.06; raw LV1 centroid **역전**(trace>1+>4+); isotonic 후처리 후 trace~4+ 모두 1.0으로 **포화** → **sensitivity 전용**, primary 부적합 |

\* PLS는 단조 위반 시 isotonic 후처리 적용 (`afb_layer2.py`).

**요약:** culture 앵커 기반 3법(polychoric / optimal / isotonic)은 **1+~4+ 간격이 grade_v1보다 압축**(1+≈0.60 vs 0.25). Ridit은 음성이 0이 아닌 **분포 위치**. PLS(CXR)는 overlap n=314·설명력 낮아 **현 코호트에서 shape 추정 불안정**.

---

### 4.2 Ridit analysis

**질문:** “이 등급이 우리 코호트 분포에서 어느 **누적 위치**인가?”

**계산 (순서형 ridit):**

\[
\text{ridit}(g) = \sum_{g' < g} p(g') + 0.5 \cdot p(g)
\]

- \(p(g)\): 등급 \(g\) 환자 비율.
- 0이 많으면 양성 등급 ridit이 **0.5~1.0에 압축**; 희귀 등급(4+)은 1에 가까움.

| 장점 | 한계 |
|------|------|
| 통계적으로 정통, 구현 즉시 가능 | **감염력·심각도**가 아닌 **빈도 기반 순위** |
| “임의 0.25 간격” 대비 투명한 대안 | 양성 등급 간 간격이 좁을 수 있음 |

**스킴명 후보:** `afb_ridit_v1`

---

### 4.3 PLS within-AFB (양성 subset)

**질문:** “X-ray 패턴상 AFB 등급 간 **거리**는?”

**왜 full cohort PLS는 안 되나**

- 0이 ~54% → 1차 잠재축이 **등급**이 아니라 **음성 vs 양성**에 잡힘.

**추천 절차**

1. Subset: **AFB > 0** (또는 trace 제외, **1+ 이상만**).
2. \(Y\): ordinal grade (1+ … 4+).
3. \(X\): frozen CXR feature (DenseNet/ViT, **학습 안 함**) **또는** D3/D4 연속값만 (영상 PLS보다 단순).
4. PLS-DA / PLS-R → **LV1**에 등급별 **centroid** 투영.
5. Centroid를 [0,1] 정규화 → \(f(g)\).
6. **단조성 검사**: \(f(4+) > f(1+)\) 등 위반 시 isotonic/ridit으로 보정.
7. **CV**: centroid·\(f\) 표는 **train fold만**으로 추정 (leakage 방지).

| 장점 | 한계 |
|------|------|
| EER이 \(w_1\) 담당 → PLS는 **shape만** | 4+ n≈58, 불안정 |
| 논문에 loadings·설명분산 제시 가능 | AFB=객담, CXR=간접 → 축이 약할 수 있음 |

**스킴명 후보:** `afb_pls_cxr_v1`, `afb_pls_culture_v1` (X=D3/D4만)

---

### 4.4 Polychoric AFB ordinal ↔ D3 / D4

**질문:** “AFB 등급이 **배양(TTP 변환값)** 과 얼마나 정렬되는가?”

- study-level **ordinal AFB** vs **연속 D3, D4** (loginv / twostep) polychoric 상관.
- 상관 구조·등급별 조건부 분포에서 **등급 간 거리** 추정.
- **영상 불필요** — 객담 내부 정합.

| 장점 | 한계 |
|------|------|
| infectivity proxy(D3/D4)와 직접 연결 | 배양 없는 study missing |
| PLS 영상 이슈 회피 | 인과 해석 주의 (공변량 관계) |

**스킴명 후보:** `afb_polychoric_culture_v1`

---

### 4.5 Optimal scaling (CATREG / optimal quantification)

**질문:** “AFB 등급 점수를 어떻게 주면 **다른 변수와의 설명력**이 최대인가?”

- 범주(등급)에 실수 점수 부여, **반복 최적화**로 \(R^2\)·상관 극대화.
- **앵커**: D3, D4, D5, PCR, frozen CXR feature 등 (ViT head와 **동시 사용 시 순환** 주의).

| 장점 | 한계 |
|------|------|
| multi-indicator infectivity 스토리와 맞음 | 등급별 n 작으면 과적합 |
| PLS와 유사하나 **점수 자체**를 직접 최적화 | 앵커 정의가 결과를 좌우 |

---

### 4.6 Isotonic regression + anchor

**질문:** “anchor(예: culture score)와 **단조**를 유지하며 등급 점수 부여”

1. 등급별 anchor **중앙값/평균** (예: D4 twostep, D3 loginv).
2. **Isotonic regression**: grade ordinal → [0,1], \(f(1+) < f(4+)\) **강제**.
3. PLS/polychoric 결과가 단조 깨면 **후처리**로도 사용.

| 장점 | 한계 |
|------|------|
| 1+ < 4+ **보장** | anchor 품질·missing에 의존 |
| 해석 단순 | “최적”이 아닌 “단조 맞춤” |

---

## 5. 해석 가이드 — “몇 % 기여?”

| 질문 | 답 주체 | 예시 |
|------|---------|------|
| AFB **채널**이 infectivity에서 몇 %? | **1층 EER** \(w_1\) | ≈ 0.15 |
| 4+가 1+보다 **채널 안에서** 얼마나 큰가? | **2층** \(f(4+)\) vs \(f(1+)\) | 0.85 vs 0.20 |
| **한 명** 4+ 환자의 스코어 기여 | \(w_1 \times f(4+)\) | (다른 head 합성 시) |
| “4+가 전체 감염력의 30%” 단일 숫자 | — | **보통 쓰지 않음** |
| 코호트 **기대 기여도** | \(\sum_g \pi(g)\, w_1\, f(g)\) | 등급 비율 \(\pi\) × 가중치 |

---

## 6. 객담 벡터 다양성 / EER 왜곡 우려

**우려 (타당함):** micro 블록에 지표를 많이 넣으면 AFB/PCR/solid/liquid이 **한 덩어리**로 묶여 effective rank·PLS가 중복 축을 나눔.

**대응**

| 규칙 | 내용 |
|------|------|
| EER 1층 | **D1~D5 다섯 head만** (확장 금지) |
| AFB 세부 | EER에 넣지 않고 **2층 \(f\)** 만 |
| 2층 입력 축소 | AFB ordinal + (선택) D3/D4 연속 |

---

## 7. 논문·실험 로드맵

### 7.1 라벨 스킴 병렬 (ablation)

| 스킴 | 2층 방법 | 용도 |
|------|----------|------|
| `afb_grade_v1` | 임상 등간격 | **baseline** |
| `afb_ridit_v1` | Ridit | 분포 기반 sensitivity |
| `afb_within_eer_v1` | PLS / polychoric / isotonic 중 **채택** | **data-driven** 주장 |

### 7.2 작업 순서 (데이터 확보 후)

1. **EER relabel** — D1~D5 \(w_k\) 재산출 (`compute_eer_relabel.py` 계열).
2. **2층 점수표** — ridit (즉시) → polychoric(D3/D4) → PLS(AFB+ only, CV).
3. **단조 검증** — 모든 \(f\) 표에 대해 severity 순서 체크.
4. **비교표** — 등급별 \(f(g)\), Spearman between schemes, ViT/RF 성능 차이 (부차).
5. **합성 score** — \(S = \sum w_k \cdot \text{head}_k\) (Phase III 프레임, D1만 \(f\) 적용).
6. **ViT** — head별 masked MSE; **score 정의는 라벨 단계에서 고정**.

### 7.3 ViT / Phase III와의 관계

| 구분 | Phase III | Infectivity_ViT v1 |
|------|-----------|---------------------|
| D1 | 이진 | **연속 \(f\)(grade)** |
| 패널 가중치 | EER (이진 head) | **동일 EER** (연속 head에 적용 가능) |
| 영상 | DenseNet frozen + RF | ViT end-to-end |
| AFB 등급 거리 | 없음 (이진) | **2층 \(f\)** 가 핵심 기여 |

---

## 8. PLS 단독(full cohort)을 쓰지 않는 이유 (요약)

| | Full cohort PLS | 2단 (EER + AFB-only) |
|--|-----------------|----------------------|
| 0 vs 양성 | 축 지배 ❌ | EER / subset 분리 ✅ |
| 절대 비중 | PLS에 섞임 | **EER** ✅ |
| 등급 shape | 불안정 | **AFB+ PLS / ridit / polychoric** ✅ |
| 논문 방어 | 약함 | “calibration + panel weight” 분리 ✅ |

---

## 9. 오픈 결정 (구현 전)

1. **Trace (0.125)** — ridit/PLS에 포함 vs 1+에 병합 vs 별도 등급 유지.
2. **2층 채택** — **primary: polychoric culture** (`afb_polychoric_culture_v1`); sensitivity: optimal / ridit / PLS. → CXR MTL 학습 타겟: [`docs/CXR_MULTITASK_SPEC.md`](docs/CXR_MULTITASK_SPEC.md) §2.2
3. **PLS \(X\)** — CXR feature vs D3/D4 only (영상 논리 vs 객담 정합).
4. **EER 입력** — v1 graded D1을 polychoric에 **ordinal**로 넣을지, 이진 collapse 후 \(w_1\)만 쓸지.
5. **CV** — \(f(g)\) 표를 fold별로 안정적인지; 고정 표 vs fold-avg.

---

## 10. 한 줄 요약

> **EER(1층)** 로 D1~D5 **패널 비중**을 정하고,  
> **AFB 등급별 점수 \(f\)(2층)** 는 AFB 축 안에서만 **Ridit / PLS(AFB+) / Polychoric(D3,D4) / Optimal scaling / Isotonic+anchor** 로 추정한다.  
> 합성 infectivity는 Phase III와 같이 \(S = \sum w_k \cdot \text{head}_k\) 이며, D1만 \(f(\text{AFB grade})\)로 연속화한다.

---

*작성: Infectivity_ViT v1 — AFB scoring design*  
*파일: `Infectivity_ViT/v1/[중요]AFB_Scoring.md`*
