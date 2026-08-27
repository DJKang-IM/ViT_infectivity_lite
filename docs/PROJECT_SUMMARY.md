# Infectivity_ViT — 프로젝트 정리 (2026-06-26)

> Phase III 감염력 실험을 **ViT end-to-end** + **연속/등급 라벨**로 확장하는 v1 작업 요약

---

## 1. 배경 및 목표

### Phase III 한계

| 항목 | Phase III (v8.00 / v7.011) |
|------|---------------------------|
| 백본 | DenseNet121@1024 (frozen) + per-head RandomForest |
| 라벨 | D1–D5 **이진 0/1** |
| 전처리 | CLAHE → 1024×1024 stretch |
| 1차 지표 | AUROC @ threshold |
| v9 macro AUROC | D1 0.805 ~ D5 0.851 (threshold만으로 specificity 한계) |

### Infectivity_ViT v1 목표

- 백본: **ViT end-to-end fine-tune** (`vit_base_patch16_224`, 256×256 + pos embed interpolate)
- 라벨: **[0, 1] 연속/등급** — 검사 원문이 가진 **정보의 연속성** 활용
- Loss: head별 **masked MSE** (미검 = missing, 해당 head 제외)
- 1차 지표: **MSE, Spearman ρ**, (보조) 이진 AUROC @ 0.5
- Phase III EER infectivity score와의 **직접 점수 비교는 v1 범위 밖**

---

## 2. 데이터 정책 (확정)

### 2.1 CRAWLFIX vs JSON — 오해 정리

| 구분 | 설명 |
|------|------|
| **CRAWLFIX 신뢰도** | **높음.** 크롤링·정제 파이프라인으로 만든 이진 라벨은 Phase III에서 검증된 기준선 |
| **불일치의 의미** | CRAWLFIX ↔ **신고데이터(공적 신고 메타)** 간 불일치. CRAWLFIX 자체가 잘못됐다는 뜻이 **아님** |
| **v1에서 JSON을 쓰는 이유** | CRAWLFIX가 **음성/양성(0/1)만** 담고 있어, AFB 등급·PCR indeterminate·배양 TTP 등 **연속 정보가 소실**됨 |
| **v1 라벨 전략** | `[FINAL] SPUTUM DATA` JSON + CT 판독 JSON에서 **원문 파싱** → 연속 라벨 생성. CRAWLFIX는 **참조/검증용**으로 유지 가능 |

### 2.2 병원 범위 — 강남성심만

| Study ID | 병원 | v1 포함 |
|----------|------|---------|
| **1xxxx** (10000–19999) | 강남성심병원 | **포함** |
| **3xxxx** (30000–39999) | 나은병원 | **폐기 (삭제)** |
| 기타 | — | 제외 |

- 코드: `10000 <= study_no < 20000` 필터 (registry, CT, sputum 파서 공통)
- **나은병원(3xxxx) 데이터는 폐기** — 재크롤링·재사용 계획 **없음** (2026-07-05 확정)

### 2.3 데이터 소스 매핑

| 용도 | 경로 | 매칭 키 |
|------|------|---------|
| **CXR 영상 (신규 주력)** | `D:\CXR Collecting Folder (download)\*.jpg` | 파일명 `{study_no}_{reg_no}_...` |
| **수집 메타/인덱스** | `D:\260626 CXR collection\*_cxr.json` | `study_no`, `collected_files[]`, `exam_date` |
| **D1–D4 라벨** | `D:\[FINAL] SPUTUM DATA\*.json` | `study_no` |
| **D5 (Cavity)** | `D:\[260611] CT Reading Collection GUI\` | `study_no` (`*_chest_ct.json`, `*_EXTERNAL.json`, `*_ct_reading.json`) |
| **신고일 anchor** | `D:\*260509*FINAL*META*.xlsx` | Study No. → 신고일자 (날짜 윈도우용) |
| **이진 참조 (선택)** | `D:\260611_TRAINING_META_CSV_CRAWLFIX.csv` | 검증·ablation 시 Phase III와 대조 |

### 2.4 영상 처리 방향 (JPEG 전환, 2026-07-05)

1. **학습 영상 = JPEG.** `D:\CXR Collecting Folder (download)`의 크롤링 JPEG가 새로 작업할 **유일한 영상 소스**.
2. **DICOM 경로 폐기**: `[260509][RAWData] CXR Active`, `[260626] ViT_Infectivity`, `260611_Active_Merged_CRAWLFIX`는 **더 이상 사용하지 않음**.
3. 수집 인덱스 `260626 CXR collection\*_cxr.json`에서 study별 JPEG 경로·검사일·anchor를 읽어 라벨과 매칭.
4. JPEG 밝기 편차는 [`PREPROCESS_JPG_BRIGHTNESS.md`](PREPROCESS_JPG_BRIGHTNESS.md) / [`JPG_PREPROCESS_PLANNED.md`](JPG_PREPROCESS_PLANNED.md) 계획대로 정합.
5. 최종 조합: **강남성심 1xxxx + JSON 라벨 + CXR JPEG**.

> CXR Multi-task 학습 스펙 전체: [`CXR_MULTITASK_SPEC.md`](CXR_MULTITASK_SPEC.md).

---

## 3. 연속 라벨 스킴 (v1)

### D1 — AFB smear (`afb_grade_v1` default)

| 원문 | 값 |
|------|-----|
| Negative / No AFB | 0.0 |
| Trace / Doubtful | 0.125 |
| 1+ / Rare | 0.25 |
| 2+ / Few | 0.50 |
| 3+ / Moderate | 0.75 |
| 4+ / Numerous | 1.0 |

Variants: `afb_binary`, `afb_ordinal_5`, `afb_raw_div4`

### D2 — TB-PCR (`pcr_soft_v1`)

| 결과 | 값 |
|------|-----|
| Negative | 0.0 |
| **Indeterminate** | **0.5** |
| Positive | 1.0 |

Variant: `pcr_binary_v9` (indeterminate → 1.0, Phase III 호환)

### D3/D4 — 고체·액체 배양 (`culture_ttp_v1`)

| 경우 | 값 |
|------|-----|
| 음성 (no growth, NTM) | 0.0 |
| TB 양성, 소요일 d | transform(d) |
| 미검 | missing |

Transform: `inv` | **`loginv` (default)** | `rank` (코호트 내 순위)

- d = 양성 보고일 − 처방일 (`>> 중간/최종검사결과` 날짜 우선)
- 고체(D3)·액체(D4) **별도 head**

### D5 — Cavity

- Phase III와 동일 **0/1**
- CT 판독 JSON, 신고일 **-3 ~ +1개월**, negation regex
- in-window CT 없음 → **missing** (CSV fallback 없음)

---

## 4. 모델·학습 (v1)

```
Chest CT DICOM
  → HU window → CLAHE (clip 0.03) → 256×256 stretch → 3ch → ImageNet norm
  → ViT-Base (fine-tune) → 768-d
  → 5 × (Linear + Sigmoid) → [0,1]^5
  → masked MSE loss
```

| 항목 | 값 |
|------|-----|
| CV | 5-fold StratifiedKFold (study-level, D4 binary stratify) |
| Early stopping | val macro MSE, patience 5 |
| Batch | 8, AMP on GPU |

---

## 5. 구현 현황

### 프로젝트 구조

```
D:\Infectivity_ViT\v1\
├── README.md
├── docs\PROJECT_SUMMARY.md          ← 이 문서
├── configs\default.yaml
├── src\
│   ├── labels\                      # JSON → 연속 라벨 CSV
│   ├── data\                        # DICOM preprocess, dataset
│   ├── models\vit_multitask.py
│   ├── train.py, eval.py
│   └── run_ablation.py
├── scripts\
│   ├── run_build_labels.ps1
│   ├── run_train_v1.ps1
│   └── run_label_ablation.ps1
└── artifacts\
    ├── labels_v1.csv
    └── labels_v1_audit.csv
```

### 라벨 빌드 1차 결과 (강남성심 1xxxx, JSON registry)

```
registry (union): 1601 studies
  sputum JSON: 1540 | CT reading: 1465
  D1 labeled: 1377 | D2: 728 | D3/D4: 1377 | D5: 1323
```

### 미완 / 다음 단계

| 항목 | 상태 |
|------|------|
| **CXR JPEG 크롤링 수집** | **진행 중** (`CXR Collecting Folder (download)`) |
| JPEG 인덱서 + JPEG loader Dataset | 신규 구현 필요 ([`CXR_MULTITASK_SPEC.md`](CXR_MULTITASK_SPEC.md) §3) |
| ~~`[260626] ViT_Infectivity` DICOM 머지·헤더~~ | **폐기** (DICOM 미사용) |
| AFB × culture label ablation (9 runs, fold 0) | 스크립트 준비됨 |
| CRAWLFIX 이진 vs JSON 연속 라벨 **대조 리포트** | 선택 (검증용) |

---

## 6. 실험 매트릭스 (계획)

1. **Baseline**: default schemes + ViT@256 + CLAHE, 5-fold
2. **Label ablation**: AFB 3 variants × culture transform 3 = 9 (fold 0 screening)
3. 상위 2–3 scheme → full 5-fold

---

## 7. Phase III 대비 변경 요약

| | Phase III | Infectivity_ViT v1 |
|---|-----------|-------------------|
| 백본 | DenseNet121 frozen | ViT E2E |
| 해상도 | 1024 | 256 |
| 라벨 | CRAWLFIX 이진 0/1 | JSON 파싱 연속 [0,1] |
| CRAWLFIX 역할 | 학습 GT | 참조·검증 (신뢰도 높음, 연속성 부족) |
| 병원 | 혼합 가능 | **강남성심 1xxxx만** |
| 감염력 점수 | EER 가중 | v1 범위 밖 (후속) |
| D1 (CXR MTL) | — | **`afb_polychoric_culture_v1`** — [`CXR_MULTITASK_SPEC.md`](CXR_MULTITASK_SPEC.md) |

---

## 8. CXR Multi-task Learning (별도 스펙)

흉부 **X-ray** 기반 5-head MTL (ViT-S / Swin-T → Hybrid / LoRA) 설계 문서:

**[`docs/CXR_MULTITASK_SPEC.md`](CXR_MULTITASK_SPEC.md)**

| 항목 | v1 (현재 코드) | CXR MTL 스펙 |
|------|----------------|--------------|
| 영상 | CT DICOM | **CXR** |
| Split | image CV | **patient 70/15/15** |
| D1 | `afb_grade_v1` | **polychoric** ([`[중요]AFB_Scoring.md`](../[중요]AFB_Scoring.md) §4.1.1) |
| Loss | masked MSE | **MSE + BCE** |
| D2, D5 | 연속/미사용 | **Binary BCE** |

D1 polychoric 점수표·근거: [`[중요]AFB_Scoring.md`](../[중요]AFB_Scoring.md), `src/labels/afb_layer2.py`

---

## 9. 빠른 실행

```powershell
cd D:\Infectivity_ViT\v1
pip install -r requirements.txt

# 연속 라벨 CSV (JSON 기반, 1xxxx only)
.\scripts\run_build_labels.ps1

# 학습 (DICOM 경로는 config 확인)
.\scripts\run_train_v1.ps1
```

---

_작성: 2026-06-26. 라벨/DICOM 경로·코호트 변경 시 이 문서와 `configs/default.yaml`을 함께 갱신._
