# Machine Learning & Econometrics: Capital Structure and Wage Prediction

Two empirical projects from the *Machine Learning & Econometrics* course in Fordham University's
MS in Quantitative Finance. Both pull their data programmatically from the source, so nothing is
downloaded by hand. Each one runs end to end in a single notebook.

| | Question | Data | Methods | Write-up |
|---|---|---|---|---|
| **1** | What determines how much debt a firm carries? | Compustat via **WRDS API**, 36,718 firm-years | Pooled OLS, industry/year and firm fixed effects, clustered SEs | [Report](reports/01_capital_structure_report.md) · [PDF](reports/01_capital_structure_report.pdf) · [Notebook](notebooks/01_capital_structure.ipynb) |
| **2** | How well can personal characteristics predict wages? | 2020 ACS via **IPUMS API**, 875,856 workers | OLS, forward stepwise, Lasso, Ridge, random forest | [Report](reports/02_wage_prediction_report.md) · [PDF](reports/02_wage_prediction_report.pdf) · [Notebook](notebooks/02_wage_prediction.ipynb) |

## Key findings

**Capital structure.** Larger firms, firms with tangible assets and firms in high-leverage
industries borrow more. Profitable and cash-rich firms borrow less, consistent with the
pecking-order theory. Pooled OLS explains 21–25% of leverage. Within-firm changes explain only
3%, because leverage is highly persistent.

![Standardized effects on leverage](reports/figures/cs_standardized_effects.png)

**Wage prediction.** A random forest gives the best out-of-sample fit (test R² 0.595 vs. 0.558 for
OLS) by capturing interactions between hours, education and occupation. Lasso and Ridge do not
improve on OLS, because with 160k observations and 154 features cross-validation chooses
essentially no penalty.

![Model comparison](reports/figures/wage_model_comparison.png)

## Repository layout

```
notebooks/
  01_capital_structure.ipynb   WRDS query -> sample -> descriptives -> OLS / FE models
  02_wage_prediction.ipynb     IPUMS extract -> sample -> 5 models -> evaluation
reports/
  01_capital_structure_report.md / .pdf
  02_wage_prediction_report.md / .pdf
  figures/                     charts produced by the notebooks
data/                          created locally; licensed data is never committed
```

## Reproducing the results

```bash
git clone https://github.com/SGhuman123/Fordham-ML-Econometrics-Project.git
cd Fordham-ML-Econometrics-Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your credentials
jupyter lab notebooks/
```

**Credentials** (both free for academic users):

- **WRDS:** requires a WRDS account with Compustat access. Set `WRDS_USERNAME` in your environment
  and store the password once with
  `python -c "import wrds; wrds.Connection().create_pgpass_file()"`.
- **IPUMS:** create an API key at <https://account.ipums.org/api_keys> and set `IPUMS_API_KEY` in
  `.env`. The extract takes a few minutes to build on the IPUMS servers.

On the first run, each notebook downloads its data to `data/raw/` and caches it for later runs.
Notebook 1 runs in under a minute, and notebook 2 in about two minutes on a laptop. All random
steps are seeded.

**Data licences.** Compustat data are licensed through WRDS, and IPUMS terms prohibit
redistributing the microdata, so `data/` is excluded from the repository. If you reuse the ACS
extract, please cite IPUMS USA (Ruggles et al., *IPUMS USA*, Minneapolis, MN: IPUMS).

## Tools

Python · pandas · scikit-learn · statsmodels · linearmodels · matplotlib · WRDS API · IPUMS API
