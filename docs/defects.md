# Разбор четырёх дефектов

## 1. Общий run для всех кандидатов
- **Симптом:** в MLflow один run `all-candidates` вместо шести отдельных; `search_runs(parentRunId=sweep)` возвращает 0.
- **Причина:** все кандидаты писали в один вложенный run, `active_run()` был общим.
- **Исправление:** каждый кандидат обёрнут в свой `mlflow.start_run(run_name=...)`, вложенность убрана.
- **Доказательство:** `make check` → проверка 1 PASS, 6 отдельных run.

## 2. Потеря артефактов после пересоздания контейнера
- **Симптом:** `make check-storage` падает: SHA256 скачанного `probe.txt` не совпадает с исходным.
- **Причина:** `--artifacts-destination=/mlartifacts` указывает внутрь контейнера, том `artifacts` объявлен, но не смонтирован.
- **Исправление:** добавлен `volumes: [artifacts:/mlartifacts]` в сервис `mlflow`.
- **Доказательство:** `make check-storage` PASS, хеш совпадает после `--force-recreate`.

## 3. Отсутствие версий train/val в runs
- **Симптом:** `make check` → проверка 2 падает на `data_train_sha256`/`data_val_sha256`.
- **Причина:** `versions = {k: v for k, v in provenance(p).items() if not k.startswith("data_")}` выкидывал хеши данных из тегов.
- **Исправление:** `versions = provenance(p)` — все ключи попадают в `tags` run.
- **Доказательство:** `make check` → проверка 2 PASS.

## 4. Обход порога по loss при регистрации
- **Симптом:** `make check` → проверка 6 падает: `promote(..., threshold=0.0)` не отклоняет кандидата.
- **Причина:** `promote()` вычислял `ratio`, но не вызывал `gate(run, threshold)`.
- **Исправление:** `ratio = gate(run, threshold)` в начале функции — при нарушении порога бросается `ValueError` до `register_model`.
- **Доказательство:** `make check` → проверка 6 PASS.