# Data report: OTU_2D ovarian ultrasound dataset

## 1. Source
- **Dataset:** OTU_2D, the 2D B-mode part of MMOTU (Multi-Modality Ovarian Tumor Ultrasound), Beijing Shijitan Hospital.
  Paper: arXiv 2207.06799. Download: https://drive.google.com/drive/folders/1c5n0fVKrM9-SZE1kacTXPt1pt844iAs1
- **Contents:** 1,469 still ultrasound images in 8 categories, with segmentation masks. The dataset has no patient IDs.
- **Our task:** binary classification. **Only high-grade serous cystadenocarcinoma counts as malignant.** The other 7
  categories are benign lesions or normal ovary.

## 2. What we have

| Category | Label | Raw | Dropped | Kept |
|---|---|---:|---:|---:|
| Chocolate cyst | benign | 336 | 30 | 306 |
| Teratoma | benign | 336 | 32 | 304 |
| Normal ovary | benign | 267 | 24 | 243 |
| Serous cystadenoma | benign | 219 | 20 | 199 |
| Mucinous cystadenoma | benign | 104 | 8 | 96 |
| Theca cell tumour | benign | 88 | 6 | 82 |
| Simple cyst | benign | 66 | 3 | 63 |
| **High-grade serous (malignant)** | **malignant** | **53** | **3** | **50** |
| **Total** | | **1,469** | **126** | **1,343** |

- **Dropped:** 110 near-duplicate frames, plus 16 near-duplicates whose labels conflicted with their twin.
- **Kept:** **1,293 benign vs. 50 malignant (3.7%)**, about 26 benign images for every malignant one.
- **Independent malignant cases:** the 50 malignant images fall into **46 visual-similarity groups**, so there are about 46
  independent malignant cases. The real number of patients may be lower, since the dataset has no patient IDs.

## 3. Image properties
- **Original sizes:** 302–1,135 px wide and 266–794 px high. Every image is letterboxed (padded, not stretched) to 224×224.
- **Image types:** mostly grayscale B-mode. Some frames carry colour Doppler, which the "colour" experiments keep.
- **On-screen overlays:** measurement tables, calipers and text are removed or inpainted. Some caliper remnants survive, and
  in 139 images the inpainted table leaves a grey box.

## 4. How it's split
- **Fixed split:** train 1,005 (36 malignant), val 162 (7), test 176 (7).
- **5-fold CV:** 266–275 images and **9–12 malignant per fold**. Folds are grouped so that visually near-identical frames never
  straddle folds. With CV, every malignant image is tested once, which gives better estimates than a test set of only 7.

## 5. Problems with the data
1. **Too few malignant images (the main issue).** With 50 positives:
   - Sensitivity can only be measured to about ±12 points. For example, 35/50 = 70% has a 95% CI of 56–81%.
   - One malignant image moves sensitivity by 2 points on CV, and by 14 points on the fixed test set.
2. **Only one cancer type.** All malignant images are high-grade serous. Borderline tumours and other cancers (clear cell,
   endometrioid, mucinous carcinoma) are absent, so the model never learns "malignant" in general.
3. **Possible shortcuts.** Grad-CAM showed the E0 model attending to the grey table box. Caliper remnants could also act as
   shortcuts, so the model might learn the machine's overlays rather than the anatomy.
4. **No patient IDs.** We approximate patient-level separation with image-similarity grouping. Different frames of the same
   patient could still end up in different folds.
5. **One hospital, still images only.** There's no device, operator or site variation, and no video context, so performance on
   other scanners is unknown.

## 6. Evidence that the data is the limiting factor
- **The models memorise the training data.** Training AUC reaches 1.000 in all 15 runs (3 models × 5 folds). Validation AUC
  peaks at 0.75–0.93 around epochs 35–60, then falls to 0.6–0.8.
- **Changing the method doesn't move the result.** Twelve or more variants (augmentation, colour, 8-class head, test-time
  augmentation, ensembles, focal loss, label smoothing, lesion crops, longer training) all land between AUC 0.79 and 0.86, mostly
  within each other's confidence intervals.
- **Results depend on which images land in the test fold.** Fold AUC ranges from 0.78 to 0.93.
- **Not yet proven.** A learning curve (training on 25/50/75/100% of the data) would confirm it. If AUC is still rising at 100%,
  more data would help.

## 7. Other public datasets
We found no public ovarian ultrasound dataset with more pathology-confirmed malignant images:
- Kaggle "MMOTU" is a re-upload of this same data.
- The newer public sets have no malignant class (normal / follicle / PCO).
- Larger malignant cohorts (for example 195 malignant patients, J Clin Med 2024) are available only on request.

## 8. Bottom line
The dataset is clean and correctly split, but it is **small and narrow in the malignant class**: 50 images of one cancer type,
from one hospital. Most of the evidence suggests this, rather than the model or training method, is what limits performance.
Results should be reported with confidence intervals and presented as a proof of concept, not a clinical tool.
