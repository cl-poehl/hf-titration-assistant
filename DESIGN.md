# HF Titration Assistant — Design & Clinical Background

This document records the clinical motivation and system design behind the
prototype. For how to run it, see [`README.md`](README.md); for a file-by-file
code map, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Background

Heart failure (HF) is a leading cause of 30-day hospital readmissions. A growing
number of "hospital-at-home" (HaH) programs manage acutely ill HF patients
outside the hospital, relying on remote monitoring of vitals, periodic labs, and
patient-reported symptoms. Monitoring is often threshold-based, which tends to
fire either too late or too often. The premise of this prototype is that a model
trained on the temporal trajectory of a patient's data can flag deterioration
earlier and more selectively than fixed thresholds — provided its output is
explainable and a clinician remains the decision-maker.

## Related work

**ML for short-term HF deterioration and readmission.** Predicting short-horizon
deterioration and 30-day readmission in heart failure from routine clinical data
is an active but genuinely hard problem. Systematic-review evidence finds
gradient-boosted trees (XGBoost [21], LightGBM [22]) and random forests to be the
most common and generally best-performing models on tabular EHR features, but discrimination is
modest and heterogeneous across cohorts (AUROC ≈ 0.51–0.93), with 30-day
readmission a comparatively difficult target [1,2,4]. Representative single-cohort
studies report best-model AUROCs of roughly 0.73–0.81 (e.g. XGBoost AUC 0.763 on
2,232 acute-HF admissions [1]; an ensemble at 0.81 [3]). HF Titration Assistant's 0.80 AUROC
on its synthetic cohort sits within this range — but, being synthetic, it reflects
internal consistency rather than clinical accuracy. Dynamic (time-varying)
prediction from home telemonitoring has reached ≈0.80 AUROC and flags emerging
risk earlier than nurse-led telesupport, albeit with sequence models rather than
trees [5].

**Remote monitoring, telemonitoring and virtual wards.** The clinical evidence
base for remote HF management is substantial but mixed. The TIM-HF2 RCT showed
structured remote patient management reduced days lost to unplanned cardiovascular
admission and all-cause mortality in selected patients [6], whereas BEAT-HF found
post-discharge telemonitoring plus coaching did not reduce readmission [7].
Implantable hemodynamic monitoring shows the same split: CHAMPION (CardioMEMS)
markedly reduced HF hospitalizations [8], while GUIDE-HF missed its primary
endpoint [9]. This heterogeneity motivates decision support that is *selective and
explainable* rather than alert-heavy. (RCT-grade evidence for hospital-at-home /
virtual-ward models specifically in HF — as distinct from telemonitoring and
implantable monitoring — remains limited.)

**Explainability.** The per-prediction explanations use SHAP [10], a game-theoretic
additive feature-attribution framework, computed with the exact polynomial-time
TreeExplainer for tree ensembles [11], which was itself validated on clinical risk
tasks. SHAP is widely used to rank predictors and produce per-patient explanations
in HF risk models [1,3].

**Guideline-directed medical therapy and titration gaps.** The rule-based GDMT
engine follows the four-pillar approach (RAASi/ARNI, beta-blocker, MRA, SGLT2i) of
the 2022 AHA/ACC/HFSA guideline [12] and the 2023 ESC focused update [13].
Real-world registries document large gaps between guideline targets and
prescribing — in CHAMP-HF, many eligible HFrEF patients were not prescribed
indicated therapy and few reached target doses [14] — which is precisely the gap
decision support of this kind aims to close.

**Data and reporting.** Development uses a synthetic generator plus the open
MIMIC-IV [15] and eICU-CRD [16] demo subsets from PhysioNet [17]. Reporting of
clinical prediction models is guided by standards such as TRIPOD+AI [18], PROBAST
for risk-of-bias [19], and attention to calibration [20]; patient-level (grouped)
data splitting is used here to avoid leakage across a patient's temporal windows.

## Clinical scope

- **Condition:** heart failure across ejection-fraction phenotypes (HFrEF,
  HFmrEF, HFpEF)
- **Setting:** acute hospital-at-home, ~30-day monitoring window
- **Prediction target:** deterioration requiring escalation within a short
  horizon (7 days for synthetic data; 3 days for the shorter ICU-derived demo
  cohorts)
- **Inputs:**
  - Vitals (weight, SpO₂, heart rate, blood pressure, respiratory rate),
    ~twice daily
  - Labs (BNP/NT-proBNP, creatinine, potassium), roughly weekly
  - Patient-reported symptoms (dyspnea, orthopnea, edema, exercise tolerance,
    medication adherence), roughly daily

## System design

```
┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│  Data layer  │──▶│ Feature engineering│──▶│  Prediction engine   │──▶│   Dashboard     │
│ parquet on   │   │ rolling stats,     │   │ XGBoost/LightGBM +   │   │ risk panel,     │
│ disk         │   │ trends, baseline   │   │ SHAP attributions,   │   │ trend charts,   │
│              │   │ deviations,        │   │ risk tiers, actions  │   │ SHAP reasoning  │
│              │   │ missingness signals│   │                      │   │                 │
└──────────────┘   └───────────────────┘   └──────────────────────┘   └─────────────────┘
```

The pipeline is deliberately file-based (parquet, no database) so the full
feature → train → serve loop is reproducible from a single clone.

## Feature engineering rationale

Features are extracted over sliding windows and are chosen to capture clinically
meaningful dynamics rather than single-timepoint values:

- **Weight velocity** — rapid weight gain is an early congestion signal.
- **Shock index** (HR / systolic BP) — a hemodynamic instability marker.
- **Rolling trends and baseline deviations** — a patient is compared against
  their own recent baseline, not a population threshold.
- **Symptom burden and reporting rate** — both the reported symptoms and the
  *rate* at which a patient reports are treated as signal.
- **Missingness indicators** — gaps in monitoring can themselves be informative.

## Validation methodology

Splits are made at the **patient level** (all temporal windows of a given patient
land in the same fold), which prevents leakage of a patient's future windows into
the training set — a common and easy-to-miss pitfall when rows are windows of a
time series. Models are evaluated with AUROC (reported with a 95% confidence interval from
1,000 patient-level bootstrap resamples), AUPRC, a calibration curve, and
per-tier sensitivity / specificity / PPV. See [`README.md`](README.md#results)
for reported numbers and their caveats.

## Risk tiers and suggested actions

Tier boundaries are defined once in `RISK_TIER_THRESHOLDS` (`src/api/predictor.py`):

| Tier | Deterioration probability | Suggested action | Urgency |
|------|---------------------------|------------------|---------|
| Low | < 15% | Continue routine monitoring | Routine |
| Medium | 15–59% | Phone check-in, review meds/fluid status | Soon |
| High | 60–79% | Alert physician, consider in-person visit + diuretic adjustment | Urgent |
| Critical | ≥ 80% | Immediate physician notification, evaluate for ED transfer | Emergent |

Every alert is paired with a SHAP explanation, and the design keeps a human in
the loop at every tier — the tool suggests, it does not decide.

## Guideline-directed medical therapy (GDMT) view

A rule-based engine surfaces GDMT titration opportunities across the four
guideline pillars (RAASi, beta-blocker, MRA, SGLT2i) with per-class safety checks
(SBP, HR, potassium, eGFR, creatinine trend, symptom burden), following the
2022 AHA/ACC/HFSA HF guideline. In this prototype the per-patient medication
state is **synthetic**, included to demonstrate the interface and safety logic
rather than to represent real prescribing.

## Regulatory considerations (aspirational)

A production system in this space would need to address the FDA clinical decision
support (CDS) exclusion criteria (human-in-the-loop, transparent reasoning, no
autonomous treatment decisions) [23], HIPAA safeguards, and — for EU deployment —
EU MDR and EU AI Act obligations. **None of these are implemented here**; they are
noted only to document what a real deployment would require.

## References

1. Zhang Y, et al. Explainable machine learning for predicting 30-day readmission in acute heart failure patients. *iScience.* 2024;27(7):110281. doi:10.1016/j.isci.2024.110281
2. Yu H, Son G-H. Machine learning-based 30-day readmission prediction models for patients with heart failure: a systematic review. *Eur J Cardiovasc Nurs.* 2024;23(7):711. doi:10.1093/eurjcn/zvae031
3. Pikatza-Huerga A, Almeida JG, Quirós C, et al. Machine learning approaches for predicting heart failure readmissions. *Postgrad Med J.* 2025;101(1202):1351–1360. doi:10.1093/postmj/qgaf102
4. Sabouri M, et al. Machine learning based readmission and mortality prediction in heart failure patients. *Sci Rep.* 2023;13:18671. doi:10.1038/s41598-023-45925-3
5. Fahimi J, et al. A vital signs telemonitoring programme improves the dynamic prediction of readmission risk in patients with heart failure. *AMIA Annu Symp Proc.* 2020;2020. PMCID: PMC8075426
6. Koehler F, Koehler K, Deckwart O, et al. Efficacy of telemedical interventional management in patients with heart failure (TIM-HF2): a randomised, controlled, parallel-group, unmasked trial. *Lancet.* 2018;392(10152):1047–1057. doi:10.1016/S0140-6736(18)31880-4
7. Ong MK, Romano PS, Edgington S, et al. Effectiveness of remote patient monitoring after discharge of hospitalized patients with heart failure (BEAT-HF). *JAMA Intern Med.* 2016;176(3):310–318. doi:10.1001/jamainternmed.2015.7712
8. Abraham WT, Adamson PB, Bourge RC, et al. Wireless pulmonary artery haemodynamic monitoring in chronic heart failure (CHAMPION): a randomised controlled trial. *Lancet.* 2011;377(9766):658–666. doi:10.1016/S0140-6736(11)60101-3
9. Lindenfeld J, Zile MR, Desai AS, et al. Haemodynamic-guided management of heart failure (GUIDE-HF): a randomised controlled trial. *Lancet.* 2021;398(10304):991–1001. doi:10.1016/S0140-6736(21)01754-2
10. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst (NeurIPS).* 2017;30:4765–4774.
11. Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell.* 2020;2(1):56–67. doi:10.1038/s42256-019-0138-9
12. Heidenreich PA, Bozkurt B, Aguilar D, et al. 2022 AHA/ACC/HFSA guideline for the management of heart failure. *Circulation.* 2022;145(18):e895–e1032. doi:10.1161/CIR.0000000000001063
13. McDonagh TA, Metra M, Adamo M, et al. 2023 focused update of the 2021 ESC guidelines for the diagnosis and treatment of acute and chronic heart failure. *Eur Heart J.* 2023;44(37):3627–3639. doi:10.1093/eurheartj/ehad195
14. Greene SJ, Butler J, Albert NM, et al. Medical therapy for heart failure with reduced ejection fraction: the CHAMP-HF registry. *J Am Coll Cardiol.* 2018;72(4):351–366. doi:10.1016/j.jacc.2018.04.070
15. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Sci Data.* 2023;10:1. doi:10.1038/s41597-022-01899-x
16. Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Sci Data.* 2018;5:180178. doi:10.1038/sdata.2018.178
17. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation.* 2000;101(23):e215–e220. doi:10.1161/01.CIR.101.23.e215
18. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378. doi:10.1136/bmj-2023-078378
19. Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. *Ann Intern Med.* 2019;170(1):51–58. doi:10.7326/M18-1376
20. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med.* 2019;17:230. doi:10.1186/s12916-019-1466-7
21. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: *Proc. 22nd ACM SIGKDD (KDD '16).* 2016:785–794. doi:10.1145/2939672.2939785
22. Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. *Adv Neural Inf Process Syst (NeurIPS).* 2017;30:3146–3154.
23. U.S. Food and Drug Administration. *Clinical Decision Support Software — Guidance for Industry and FDA Staff.* 2022.

> **Verification note.** The bibliographic details of refs 1–11 (authors, venue,
> year, DOI) and their associated claims were cross-checked against the primary
> sources during preparation of this review. Refs 12–23 are well-established
> primary sources cited from the publisher of record but were not independently
> re-verified here — confirm against the DOI before reuse in formal work. Reported
> AUROCs in refs 1–5 are single-cohort results, not field-wide benchmarks; the
> 0.51–0.93 spread reflects genuine study heterogeneity.
