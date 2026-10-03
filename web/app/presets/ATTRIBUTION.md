# Example image pairs: sources and licenses

These images are not covered by the repository's MIT license; each keeps its source's terms.
All were converted (8-bit, cropped, resampled; JPEG only for the Landsat and histology photographs, never for IN718) by `scripts/make_presets.py`.

## IN718 · EBSD vs BSE (`in718-65`, `in718-85`) — public domain (NIST)

Schwalbach, E. J., Chapman, M. G., Shah, M. N., Uchic, M. D., Hrabe, N., Kafka, O., Moser, N., Lane, B.,
Carson, R., Belak, J., Levine, L. E. (2024). *AM Bench 2022: IN718 Serial Sectioning and X-ray Computed
Tomography Measurement Data.* National Institute of Standards and Technology. https://doi.org/10.18434/mds2-2767

Moving: EBSD IPF-Z map of the section (from the indexed `.ctf`). Fixed: the section's BSE_2 image as NIST
warped and registered it, contrast-stretched to 8 bits. NIST data are not subject to copyright in the
United States; see https://www.nist.gov/open/license.

## Landsat 8 · flood vs dry season (`landsat-*`) — public domain (NASA / USGS)

*Floods Swamp Southern Thailand*, NASA Earth Observatory images by Joshua Stevens, using Landsat data
from the U.S. Geological Survey (Landsat 8 OLI, 9 January 2017 and 2 February 2014).
https://science.nasa.gov/earth/earth-observatory/floods-swamp-southern-thailand-89445/

Fixed: a crop of the 2014 scene. Moving: the 2017 scene resampled through a known rotation, scale and
shift (the preset's ground truth).

## Aerial, histology, cells (`mida-*`) — CC BY 4.0

Lu, J., Öfverstedt, J., Lindblad, J., Sladoje, N. (2021). *Datasets for Evaluation of Multimodal Image
Registration* (v1.2.0). Zenodo. https://doi.org/10.5281/zenodo.5557568 — licensed under
https://creativecommons.org/licenses/by/4.0/

- Aerial (Zurich, QuickBird): near-infrared (moving, transformed) vs RGB (fixed, reference), patches zh15_01_01, zh9_02_02.
- Histology (Eliceiri): second-harmonic generation (moving, transformed) vs bright field (fixed), core 1B_A1; re-encoded as JPEG.
- Cells (Balvan): DU145 modality A (moving, transformed) vs modality B (fixed), patch DU145_Fluo_4_f45_02_02.

All from transformation level 4; ground truth from the dataset's `info_test.csv`.
