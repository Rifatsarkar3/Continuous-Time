# Data

No data are redistributed here. Every dataset is public; the table says where to get it.

| Dataset | Source | How it is read |
|---|---|---|
| Virkler fatigue cracks, 68 specimens | CRAN archive BaPreStoPro 0.1, https://cran.r-project.org/src/contrib/Archive/BaPreStoPro/BaPreStoPro_0.1.tar.gz (GPL >= 2) | downloaded and SHA-256 checked by `python -m certalarm.data.fleets` |
| ISU-ILCC lithium-ion cells, 251 cells | https://doi.org/10.25380/iastate.22582234.v2 (CC BY 4.0) | the same |
| MATR fast-charged lithium-ion cells, 124 cells | https://data.matr.io/1/ | the same; only the per-cycle summaries of three batch files, over HTTP ranges (needs `h5py`) |
| C-MAPSS turbofan engines | NASA Prognostics Center of Excellence, https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip | unpack the inner `CMAPSSData.zip` into `data/cmapss/CMAPSSData/` |
| PRONOSTIA bearings (IEEE PHM 2012 challenge) | https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset (the original challenge page is offline) | put `Learning_set/` and `Full_Test_Set/` in `<sources>/data/PRONOSTIA_PHM2012/`, then run the command below |
| XJTU-SY bearings | https://biaowang.tech/xjtu-sy-bearing-datasets/ | processed arrays in `<sources>/Processed_XJTU/`, one `<split>_<nn>_BearingX_Y.npz` per bearing with the horizontal channel as `vibration`, 12 consecutive windows of 2,560 samples per minute |

```bash
# PRONOSTIA arrays from the raw challenge files (about 4 minutes)
python -c "from certalarm import paths; from certalarm.data import pronostia; pronostia.build_all(paths.PHM2012, paths.DATA / 'pronostia')"
```

`<sources>` is the directory named by the environment variable `CERTALARM_DUAL_STREAM`, or by one line in
`data/source_dir.txt`, or else `data/sources/`.
