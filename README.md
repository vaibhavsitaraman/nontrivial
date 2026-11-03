# nontrivial

Selected as one of 137 fellow from among 4k+ applicants. Research direction:

Flow matching as a way to learn the joint return distribution for hospital surge planning. Two regions (hub and spoke), six bed types, and a mutual-aid action. The question is whether modeling the reward dimensions together is better than those models that predict marginals and reconstruct the joint distribution.

## Code

| Folder | What it is |
|---|---|
| `src/nets` | Velocity networks, distributional TD, and the baselines (independent, floq, sliced Cramer) |
| `src/envs` | Toy chains, the hospital regime/reward world, and the tabular CVaR reference |
| `src/checks` | Short checks: plain flow matching, then 1d / 2d / correlated returns, then the tiny control MDP |
| `src/runs` | The long hospital runs, plus fetching and fitting the New Jersey hospital data |

`src/paths.py` points at `checkpoints/`, `plots/`, and `data/`.

```
python src/checks/check_flow.py
python src/runs/train_joint.py
python src/runs/train_indep.py
python src/runs/train_baselines.py
python src/runs/fetch.py
python src/runs/fit.py
python src/runs/train_nj.py
```

## Charts

| File | What it shows |
|---|---|
| `plots/loss.png` | Training loss for the hospital methods |
| `plots/compare.png` | Per-state CVaR: tabular reference against the learned policies |
| `plots/nj_loss.png` | Loss for the joint model after the New Jersey fit |
| `plots/nj_compare.png` | Same comparison on the New Jersey-fitted environment |

Those first two figures are the ones embedded in `docs/report.tex`. Longer notes are in `docs/handoff.md`.

## Saved stuff

`checkpoints/` has the trained weights (`joint.pt`, `indep_0.pt`–`indep_5.pt`, `floq.pt`, `sliced.pt`, `nj.pt`) and the tabular references (`reference.pkl`, `nj_reference.pkl`).

`data/` has the New Jersey pull: `nj_raw.csv`, `nj_calib.json`, `nj_params.json`, and `nj_summary.txt`. On the fitted run the learned policy matched the tabular action on all 7 stable states.
