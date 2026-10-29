# Raw data (not tracked in git)

Source: Chen, D. (2012), Online Retail II, UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/502/online+retail+ii
DOI: 10.24432/C5CG6D. License: CC BY 4.0.

`retailcast prepare` downloads the archive and processes both workbook sheets.
The included processed panel is sufficient to reproduce modelling and reports
without downloading the 43.5 MiB workbook. Use `retailcast run --refresh` for
full raw-to-report reproduction.

Workbook used in this run:
- Filename: `online_retail_II.xlsx`
- SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`
- Size: 45,622,278 bytes.

The original UCI transfer was slow. For this run an unmodified workbook mirror
was retrieved from
https://raw.githubusercontent.com/KristerMartinez/online_retail_ii/main/online_retail_II.xlsx
and its first 4,128,768 bytes were byte-identical to the partial original UCI
download. The full original UCI transfer was not completed; full-file equality
was not independently verified. The actual workbook hash and transport origin
are recorded in provenance. Both sheets, schema, date range and total row count
are validated against the dataset description. The mirror supplies data only;
all project code and methodology were implemented in this project.
