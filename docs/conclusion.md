# Единый вывод A + B

## A. Loss и выбор кандидата

6 runs, `base_val_loss = 3.4273`, бюджет 4 шага × train 32 / val 8 / batch 1×2.
Победитель — `all-r8-high` (run 0480036a…), `final_val_loss = 2.6117`, ratio **0.762** ≤ 1.0.
Полный список — `docs/comparison.csv`.

## B. Оценка качества

Baseline: `Qwen/Qwen3-0.6B` (revision в `evaluation/baseline.json`).
Candidate: `my-qwen-adapter@candidate`.

| | Baseline | Candidate |
|---|---:|---:|
| overall accuracy | <...> | <...> |
| overall format_rate | <...> | <...> |

Срезы: `cardiology`, `pulmonology`, `endocrinology`, `gastro`, `nephrology` — n=10 каждый.

Diff: improved `<...>`, regressed `<...>` (`metrics/eval-diff.json`).

## Behavioral

- format: <PASS/FAIL>
- paraphrase: <PASS/FAIL>
- refusal: <PASS/FAIL>
- `make check-negative` → PASS

## Решение

<Принято / Не принято>. Причина: <...>.
Champion URI: `<...>` / не назначен.
Eval run: `<...>`.

## Известные ограничения

На Windows `mlflow.pyfunc.load_model` не справляется с backslash в `MLmodel` (проверки 4 и 7 `make check`). На Linux 9/9.
