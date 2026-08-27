# ViT_infectivity_lite

Full corpus available under NDA for research collaboration / employment. Contact: drkangim@naver.com


**Public lite release** of TB CAD / infectivity research code.

| | |
|--|--|
| Code | Full pipeline source (sanitized paths; no PHI) |
| JSON samples | **3** synthetic (`data/samples/`) |
| Preview images | **10** random 256px coarse (`data/coarse_256/`) |
| Full data / DICOM | **Not included** — gated: [DJKang-IM/ViT_infectivity](https://github.com/DJKang-IM/ViT_infectivity) |

See [`SERVING_RULES.md`](SERVING_RULES.md). Copied code files: ~83.

---

**CC BY-NC-SA 4.0, Non-commercial, Citation required. Full version gated — request access via GitHub Issues.**

**This is a coarse preview. For research collaboration contact the lite repo Issues page.**

---

# Infectivity_ViT v1

Chest CT 기반 TB 감염력 proxy(D1–D5) 예측을 **Vision Transformer end-to-end fine-tune** + **연속/등급 라벨 [0, 1]** 로 재실험하는 프로젝트입니다.

README에 전체 정리: [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md)  
CXR Multi-task Learning 스펙: [`docs/CXR_MULTITASK_SPEC.md`](docs/CXR_MULTITASK_SPEC.md)

---

## Phase III Production Baseline (v8.00 / v7.011)

| 항목 | 사양 |
|------|------|
| 입력 | Chest CT DICOM (study 단위) |
| 전처리 | HU window (DICOM WC/WW or 0.5–99.5 percentile) → CLAHE `clip_limit=0.03` → **1024×1024 stretch** → 3ch repeat → ImageNet norm |
| 백본 | **DenseNet121** (ImageNet pretrained, **frozen**) |
| 분류기 | **RandomForest** × 5 heads, 600 trees, per-head `class_weight` |
| 라벨 | D1–D5 **이진 0/1**, 미검 = **missing (-1)**, head별 exclude |
| CV | 5× stratified 70/15/15 (v9: 5-fold StratifiedKFold) |
| Threshold | v9 권고: D1=0.41, D2=0.51, D3=0.51, D4=0.65, D5=0.28 |
| v9 macro AUROC | D1 0.805, D2 0.845, D3 0.808, D4 0.814, D5 0.851 |

### Phase III 라벨 규칙 (이진)

| Head | 의미 | 이진 규칙 |
|------|------|-----------|
| D1 | AFB smear (도말) | 음성=0, 1+~4+/trace=1, 미검=missing |
| D2 | TB-PCR | 음성=0, 양성/indeterminate=1 |
| D3 | Solid culture (고체 배양) | NTM growth=0, TB isolated=1 |
| D4 | Liquid culture (액체 배양) | 동일 |
| D5 | Cavity (흉부 판독) | CT/X-ray 키워드 0/1 |

### Infectivity score (Phase III)

EER 가중 합 (v7.011/v8.00 RF 확률 × threshold):

- D5 Cavity: 0.3249
- D3 Solid: 0.1810
- D1 AFB: 0.1742
- D4 Liquid: 0.1654
- D2 TB-PCR: 0.1544

---

## Infectivity_ViT v1 변경점

| 항목 | Phase III | v1 |
|------|-----------|-----|
| 백본 | DenseNet121 frozen | **ViT end-to-end fine-tune** (`vit_base_patch16_224`) |
| 해상도 | 1024 | **256** (+ pos embed interpolate) |
| 분류기 | RF | **6× sigmoid linear head** (D1–D5, D7) |
| 라벨 | 0/1 | **[0, 1] 연속/등급** |
| Loss | BCE (RF) | **masked MSE** (missing head 제외) |
| 1차 지표 | AUROC @ threshold | **MSE, Spearman, (보조) 이진 AUROC** |

> v1에서는 라벨 스케일이 Phase III와 다르므로 **EER infectivity score 직접 비교는 범위 밖**입니다.

---

## 라벨 스킴 (v1 확정)

| Head | 의미 | v1 규칙 |
|------|------|---------|
| **D1** | AFB smear | 등급 스케일 (아래 표) |
| **D2** | TB-PCR | soft: 0 / 0.5 / 1 |
| **D3** | 고체 배양 | 음성 0, 양성 TTP → **loginv(days)** |
| **D4** | 액체 배양 | 음성 0, 양성 TTP → **twostep-C** (아래) |
| **D5** | Cavity | 0/1 (CT 판독) |
| **D6** | NTM | Phase III convention **reserved** — v1 미학습 |
| **D7** | RIF resistance PCR (Expert) | D2와 **동일** soft rule (0 / 0.5 / 1) |

### D1 AFB — `afb_grade_v1`

| 원문 | 값 |
|------|-----|
| No AFB seen / Negative for AFB | 0.0 |
| Doubtful / Trace | **0.125** |
| Rare (1+) | 0.25 |
| Few (2+) | 0.50 |
| Moderate (3+) | 0.75 |
| Numerous (4+) | 1.0 |
| **Positive for AFB** | **1.0** |

### D2 TB-PCR / D7 RIF-PCR — `pcr_soft_v1`

| 결과 | 값 |
|------|-----|
| Negative | 0.0 |
| Indeterminate | **0.5** |
| Positive | 1.0 |

- **D2**: `TB PCR` / `Mycobacterium tuberculosis PCR` (객담)
- **D7**: `Rifampin resistance PCR` 등 (객담) — TB-PCR과 별 블록
- FINAL JSON에서 TB-PCR indeterminate는 **0건**; RIF-PCR indeterminate는 **있음** → D7에서 0.5가 실제로 사용됨

### D3 고체 배양 — `loginv`

```
score = 1 / log(1 + days)    # days = 양성보고일 - 처방일, clipped to [0,1]
```

### D4 액체 배양 — `twostep` (option **C**, v1 채택)

```
if days <= 3:  score = 1.0
else:           score = loginv(days)
```

#### 액체 배양 transform 후보 (기록용)

| 옵션 | 공식 | 특징 |
|------|------|------|
| **A. rank** | 양성 코호트 내 TTP 순위 percentile | 분포 균등, 절대일 해석 약함 |
| **B. loginv + d_eff** | loginv(max(days, 7)) | 1–6일 cap, 7일 이상에서 차별 |
| **C. twostep** ✓ | d≤3 → 1.0, else loginv(d) | **v1 채택** — 빠른 양성=고득점, 느린 양성만 연속 |
| **D. inv** | 1/days | 너무 가파름, 비추 |

고체/액체는 **head별 transform 분리** (D3≠D4).

### D5 Cavity

Phase III와 동일 **0/1**. CT 판독, 신고일 -3 ~ +1개월.

---

## 데이터 (읽기 전용)

| 경로 | 용도 |
|------|------|
| `D:\[260509][RAWData] CXR_Active Image` | DICOM **원본** (헤더 미작업) |
| `D:\[260626] ViT_Infectivity` | DICOM **작업본** (헤더·머지 후, 예정) |
| `D:\260611_Active_Merged_CRAWLFIX` | DICOM 임시 (config `dicom_dir`) |
| `D:\[FINAL] SPUTUM DATA` | **D1–D4, D7** sputum JSON |
| `D:\[260611] CT Reading Collection GUI` | **D5** CT 판독 JSON (`*_chest_ct.json`, `*_EXTERNAL.json`) |
| `D:\*260509*FINAL*META*.xlsx` | 신고일자 anchor (날짜 윈도우용만) |

### 라벨 소스 정책 (v1)

- **CRAWLFIX**: 이진(0/1) 라벨로 **신뢰도 높음**. 다만 AFB 등급·PCR indeterminate·배양 TTP 등 **연속 정보가 없음**
- **v1 GT**: `[FINAL] SPUTUM DATA` + CT 판독 JSON에서 **원문 파싱** → 연속 라벨
- **불일치 맥락**: CRAWLFIX ↔ 신고데이터 간 이슈이지, CRAWLFIX 자체 품질 문제가 아님
- **병원 범위**: **강남성심 1xxxx만** (나은 3xxxx 제외)
- 상세: [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md)

---

## 빠른 시작

> **CUDA PyTorch**: 시스템 기본 `python`(3.14)은 CPU 전용 wheel만 있습니다.  
> 이 프로젝트는 **Python 3.11 + cu124 venv** (`.venv`)를 쓰며, `scripts\*.ps1`이 자동으로 사용합니다.

```powershell
cd D:\Infectivity_ViT\v1

# 최초 1회: CUDA venv 생성 (RTX 4070 / cu124)
.\scripts\setup_venv.ps1

# (선택) 터미널에서 직접 python 쓸 때
. .\activate.ps1

# 1) 연속 라벨 CSV 생성
.\scripts\run_build_labels.ps1

# 2) 학습 — 1 fold 테스트
.\scripts\run_train_v1.ps1   # 또는 fold 0만:
# .\.venv\Scripts\python.exe src\train.py --config configs\default.yaml --tag v1_fold0 --fold 0

# 3) 라벨 ablation screening (fold 0 only)
.\scripts\run_label_ablation.ps1
```

---

## 프로젝트 구조

```
v1/
├── README.md
├── configs/default.yaml
├── src/
│   ├── labels/          # build_graded_labels, encoders, parsers
│   ├── data/            # preprocess, dicom_dataset
│   ├── models/          # vit_multitask
│   ├── train.py
│   └── eval.py
├── scripts/
└── artifacts/           # labels CSV, metrics JSON, checkpoints
```

---

## 실험 매트릭스 (v1)

1. **Baseline**: default schemes + ViT@256 + CLAHE, full 5-fold
2. **Label ablation**: AFB 3 variants × culture transform 3 variants = 9 runs (fold 0 screening)
3. Best 2–3 schemes → full 5-fold (수동 선정)

결과: `artifacts/v1_{tag}/metrics.json`

---

## v1 범위 밖 (후속)

- EER 가중 infectivity score 재정의
- Multi-slice / multi-view
- 384+ 해상도 ViT
- 임상 변수 fusion
