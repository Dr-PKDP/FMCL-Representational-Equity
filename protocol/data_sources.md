# Data sources

All data sets are public. Downloads are in `data_acquisition/`; preprocessing is in `preprocessing/`.

## PTB-XL v1.0.3 (Task A development cohort)

PhysioNet. 21,799 records of 18,869 patients; 100 Hz and 500 Hz waveforms (43,598 `.hea` files; every `.dat` is 24,000 bytes at 100 Hz and 120,000 bytes at 500 Hz). `strat_fold` is patient-stratified (no patient in more than one fold; 2,173 to 2,198 records per fold). `sex`: 0 = male, 1 = female. `age` 300 denotes above 89 (293 records) and is kept in the oldest stratum.

Label counts (records): NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, HYP 2,649. Age 65+: 9,489 records (2,896 CD-positive); age 75+: 4,591 (1,583 CD-positive). Task A label: superclass CD.

Cache: per-record, per-lead z-score in float64, lead order I, II, III, aVR, aVL, aVF, V1 to V6; one flat lead (ecg_id 12722).

## CODE-15% (Task A external cohort)

Zenodo record 4916206. 345,779 exams of 233,770 patients in 18 archives (HDF5 inside); labels 1dAVb, RBBB, LBBB, SB, ST, AF. Conduction-disturbance analogue = 1dAVb, RBBB or LBBB (prevalence 2.9% under 65, 9.0% at 65 to 74, 13.6% at 75 to 89). Traces are 400 Hz with zero padding; the measured spans are 2,934 samples (201,084 traces) and 4,096 samples (130,763 traces). 343,809 exams are usable (670 all-zero and 1,300 spanning under 5 s are excluded); results can be stratified by `span_samples_400hz`.

## Chapman-Shaoxing and Ningbo (Task A replication cohort)

PhysioNet `ecg-arrhythmia` 1.0.0, combined release: 45,152 records, 500 Hz, 5,000 samples, no patient identifiers. All `SHA256SUMS` entries verify. Label from SNOMED codes in `#Dx:`: 1AVB 270492004 (1,140), RBBB 59118001 (649), LBBB 164909002 (240); union 1,957 records (4.3%). Age 0 (247 records) and missing age (55) are excluded from age strata. Records JS01052 and JS23074 have malformed headers and are excluded (45,150 usable).

Site is inferred from numbering and counts: JS00001 to JS10646 Shaoxing (10,247 records, 10,246 usable), above Ningbo (34,905 records, 34,904 usable). Prevalence by age (under 65 / 65-74 / 75-89): all 2.2 / 5.4 / 9.2%; Shaoxing 4.1 / 9.4 / 18.1%; Ningbo 1.6 / 4.1 / 6.6%. The file `Diagnostics.xlsx` does not map to the JS records and is not used.

## PAD-UFES-20 (Task B development cohort)

Mendeley Data, DOI 10.17632/zr7vgbcyr2.1 (`zr7vgbcyr2-1.zip`, 3,593,344,681 bytes: `images/imgs_part_1..3.zip`, `metadata.csv`). Downloaded manually in a browser; `data_acquisition/verify_pad_ufes20.py` checks the 2,298 PNGs. `metadata.csv` md5 `1c1964f098cc3fa62761c9de17228158` (2,298 rows, 1,373 patients). A Kaggle mirror (`mahdavi1202/skin-cancer`) is byte-identical for all images.

Malignant or pre-malignant lesions (BCC, SCC, MEL, ACK) are 74% to 89% of each age band (biopsy-referral cohort). Patients 65+: 561 (472 positive); 75+: 245 (210 positive). Images with a patient aged 65+: 1,007 (528 malignant); 75+: 456 (262 malignant). Gender and Fitzpatrick type are missing for the same 804 records (35%). Folds and labels: `taskB_protocol.md`.

## Fitzpatrick17k (Task B secondary check)

Annotations from `github.com/mattgroh/fitzpatrick17k` (`fitzpatrick17k.csv`: 16,577 rows; no age or sex field). `md5hash` is the md5 of the image bytes and is the integrity gate for every file (`scrape_fitzpatrick17k.py`, `quarantine_fitzpatrick.py`). Verified set: 3,887 images (23.4%), all from one atlas (atlasdermatologico.com.br); 18 URLs of that host no longer resolve, and the images of the second host (dermaamin.com) could not be retrieved. Retrieval by skin type: I 161 of 2,947, II 658 of 4,808, III 996 of 3,308, IV 1,082 of 2,781, V 636 of 1,533, VI 201 of 635, unlabelled 153 of 565. By label: benign 17.1%, malignant 31.0%, non-neoplastic 23.2%. The verified set is not representative; its skew toward darker types is a limitation of the secondary check. A published audit (arXiv 2401.14497) reports duplicates and label errors in the dataset.

## HINTS 7

National Cancer Institute, `hints7_public.sav` (2024 public-use file). Participation models: `participation_model.md`.
