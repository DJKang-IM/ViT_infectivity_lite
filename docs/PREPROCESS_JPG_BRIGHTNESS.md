# JPG CXR 밝기·톤 전처리 (보류 — 데이터 확보 후 적용)

> **상태:** 문서만 작성. 구현·실험은 DB 확보 및 해상도/배치 optimal 탐색 이후 진행.

## 배경

- 수집 데이터: **JPG** (DICOM HU/window 없음 — 이미 display-rendered)
- 문제: 장마다 **어두운 window / 밝은 window** 혼재
- 목표: 코호트 전체가 **비슷한 톤**으로 ViT 입력에 들어가게

## 하고 싶은 일 (직관)

| 입력 | 처리 방향 |
|------|-----------|
| 어두운 장 | 히스토그램을 **펼쳐서** 밝게 |
| 밝은 장 | 상위 percentile을 **눌러서** 어둡게 |
| 공통 | 최종적으로 **reference 코호트와 비슷한 톤** |

## 제안 파이프라인 (1차 — 모델 없음)

```
JPG → grayscale [0,1]
  → 2–98% percentile clip + stretch
  → reference histogram matching (중간 노출 reference 10~20장)
  → CLAHE (clip_limit 0.02~0.04, ablation)
  → resize (해상도는 별도 optimal 탐색)
  → 3ch repeat + ImageNet norm (ViT pretrained)
```

### Reference 선정 (데이터 모을 때)

- 어두움 / 중간 / 밝음 각 ~5장 스크린샷 보관
- **중간 그룹**에서 reference 10장 고정
- 가능하면 **lung mask 안** 픽셀만 matching (MFBL / XM-pipeline 계열)

## 연구 참고 (CXR homogenization)

| 방법 | 출처 | 비고 |
|------|------|------|
| MFB / MFBL standardization | Kim et al., JMI 2023 | multi-site CXR, lung mask 버전 권장 |
| XM-pipeline | Eur Radiol Exp 2023 | histogram stretch + 폐 영역 기준 |
| HyFusion | 2025 | frequency–spatial 혼합 (DICOM 멀티소스) |

**공식 pip 패키지 없음.** reference histogram + CLAHE가 방어·재현 모두 쉬움.

## DL enhancer (2차 — 필요 시만)

- Zero-DCE 등 범용 exposure net: CXR 검증 부족, artifact 리스크
- CycleGAN (밝↔어두): 데이터·학습 부담
- **우선순위 낮음** — 1차 파이프라인 ablation 후 결정

## 현재 v1 DICOM 파이프라인 (`src/data/preprocess.py`)

- HU → WC/WW 또는 percentile window → CLAHE 0.03 → resize
- JPG 수집본 적용 시 **별도 loader 분기** 필요 (구현 보류)

## 우선순위 (합의)

1. **DB 폭넓게 확보** (CXR collection GUI)
2. **해상도 × 배치** 2-track / 3-track ablation → optimal setting
3. 그다음 **본 문서 전처리 + augmentation** ablation

### 해상도·배치 후보 (RTX 4070 12GB, ViT-base)

| track | resolution | batch |
|-------|------------|-------|
| A (baseline) | 256 | 8 |
| B | 384 | 4 |
| C | 448~512 | 1~2 (+ grad accum) |

## CV / 학습 정책 (v1 확정)

- **CV split: image** (Phase III DenseNet+RF와 동일) — `cv_split_unit: image` in `configs/default.yaml`
- **슬라이스: all** (`slice_mode: all`) — study당 전체 JPG/슬라이스 학습
- study split은 리뷰어 대응용으로만 별도 실험 (`phase3_rf_study_split.py`); 기본 파이프라인 아님
