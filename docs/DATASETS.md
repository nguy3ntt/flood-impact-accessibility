# Dataset registry and acquisition plan

Documentation checked 2026-10-01. Candidate status is not proof of live download, licence clearance or case-study suitability. Verify each file before use and record the outcome in the source manifest.

| Source | Intended use | Feasibility/rights checks |
|---|---|---|
| [Sen1Floods11 repository](https://github.com/cloudtostreet/Sen1Floods11) and [layout docs](https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md) | Primary water-mapping benchmark: paired radar/optical, manual labels, permanent-water context | Verify v1.1 listing, event splits, label values, band units, acquisition offset and dataset licence separately from code |
| [NASA HLS](https://hls.gsfc.nasa.gov/) | Optional harmonised optical product for compatible foundation-model case study | Verify area/date/cloud coverage, access credentials, reflectance conventions, quality masks and historical reference labels |
| [Prithvi optical model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | Optional pretrained encoder, not a labelled dataset | Pin model revision and terms; six optical channels and product preprocessing require audit; no native radar assumption |
| [Queensland roads and tracks](https://www.data.qld.gov.au/dataset/queensland-roads-and-tracks) | Preferred local road geometry | Check coverage, topology/direction fields, snapshot date, grade separation and current attribution terms |
| [OpenStreetMap licence](https://www.openstreetmap.org/copyright) | Alternative/complementary network and facility context | ODbL attribution and derivative-database requirements; verify temporal completeness and extraction terms |
| Official facility inventory and independent event evidence, source to be selected | Hospital locations and retrospective validation | Confirm facility types/coordinates/date and actual mapped-flood or closure evidence; no fabricated download URL |

Sen1Floods11 documentation describes manual water/non-water/nodata labels and separate weak labels. Keep weak labels distinct. Its README contains an inconsistent bucket spelling; layout docs and example commands identify `sen1floods11`. Confirm the actual listing before acquiring a pilot; do not issue a full-archive download from unverified prose.

The benchmark optical imagery is not automatically equivalent to the surface-reflectance product used to pretrain an optical foundation model. Spectral mapping, reflectance level, resolution, reprojection and quality masking need a justified transform or a clearly disclosed transfer experiment.

## Acquisition order
1. Source metadata/licence and split files; list candidate files and estimate bytes.
2. Small paired manual-label pilot with event metadata and quality context.
3. Small graph/facility/origin area and independent historical-event evidence.
4. Full approved benchmark subset once M1/M2 gates pass.
5. Optional weak labels or model weights only after corresponding experiment approval/gate.

No automatic large download exists in this scaffold. Keep source URLs, access dates, checksums and redistribution status even for local-only data. Reference matching must be geographically and temporally supported; contemporary roads over old floods require a visible retrospective-mismatch disclosure.
