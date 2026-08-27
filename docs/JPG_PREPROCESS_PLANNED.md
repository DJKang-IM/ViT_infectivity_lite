# JPG CXR 밝기 정규화 전처리 (보류 — 데이터 확보 후 적용)

> **상태:** 문서만 보관. 구현·실험은 DB 확보 및 해상도/배치 optimal setting 확정 **이후**.
>
> 수집 데이터: **JPG** (DICOM HU/window 없음 — display-rendered pixel)

---

## 목표 (직관)

| 입력 유형 | 처리 방향 |
|-----------|-----------|
| 어두운 장 | 히스토그램을 **펼쳐** 전체적으로 밝게 |
| 밝은 장 | 상위 percentile을 **눌러** 과노출 완화 |
| 공통 | 코호트 **reference**와 비슷한 톤으로 homogenize |

Phase III DICOM 파이프라인의 `window → CLAHE`에 대응하는 JPG 쪽 단계.

---

## 제안 파이프라인 (1차 — 모델 없음)

```
JPG load (grayscale)
  → [0, 1] float
  → percentile clip (예: 2–98%) + stretch
  → reference histogram matching (코호트 내 “적정 노출” 10–20장)
  → CLAHE (clip_limit 0.02–0.04, Phase III와 동일 계열 0.03 시작)
  → resize (ViT 해상도)
  → 3ch repeat + ImageNet norm (pretrained ViT 유지 시)
```

### Reference 선정 (데이터 모을 때)

- 어두운 / 중간 / 밝은 샘플 각 5장 정도 스크린샷 보관
- **중간 노출** 그룹에서 reference 10장 선정
- 가능하면 **폐 mask 안** 픽셀만 histogram 통계 (lungmask / segmentation)

---

## 2차 옵션 (1차 부족할 때)

| 방법 | 출처/비고 |
|------|-----------|
| 폐 영역만 통계·matching | XM-pipeline, MFBL (Kim et al. JMI 2023) |
| MFB multi-frequency standardization | multi-institutional homogenization |
| auto-gamma (median luminance → target) | 단순 monotone curve |
| DL exposure (Zero-DCE 등) | CXR 비전용, artifact 리스크 — 최후순위 |

---

## 구현 시 터치할 코드

- `src/data/preprocess.py` — `load_tensor_from_jpg()`, reference histogram util
- `configs/default.yaml` — `input_format: jpg`, `clahe_*`, `hist_match_ref_dir`
- `src/data/dicom_dataset.py` — `.jpg` / `.jpeg` index 경로

---

## Augmentation

전처리 homogenization **확정 후** 별도 단계로:

- gamma / brightness jitter (train only)
- Phase III XM-pipeline식 contrast·noise aug 참고

---

## 참고 문헌 (research)

- Kim et al., homogenization multi-institutional CXR, *JMI* 2023 (MFB / MFBL)
- XM-pipeline, *European Radiology Experimental* 2023 (histogram + lung-region)
- HyFusion (2025) — frequency–spatial normalization (DICOM 멀티소스)

---

*작성: Infectivity_ViT v1 — 전처리 논의 보관용*
