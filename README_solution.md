# Avito — candidate generation услуг

Исследование на реальных Parquet: до 50 объявлений на запрос, метрика macro Recall@50.
Основная логика находится в семи ноутбуках. `src/` содержит только повторяемые операции:
нормализацию, текстовые индексы, признаки пар, метрику, RRF и контракт CSV.

## Результат

Независимый локальный holdout: **Recall@50 = 0.80754 cold / 0.81098 warm**, по 1 000 запросов.
Выбран CatBoost на 300 деревьев поверх hybrid-кандидатов. Это не leaderboard score.
Подробности, доверительные интервалы и ограничения — в `reports/experiment_summary.md`.

## Как запустить

Python 3.12. Рекомендуется 32–48 GB RAM; для E5 желательно CUDA или Apple Silicon MPS.
CPU поддерживается, кодирование полного каталога может занять часы.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AVITO_DATA_DIR="$PWD/dataset"
python scripts/run_notebooks.py
python scripts/validate_answer.py outputs/submissions/answer.csv
```

В `dataset/` нужны исходные `train.parquet`, `benchmark_queries.parquet`, `benchmark_items.parquet`.
Их не меняем. Полный ZIP содержит эти три файла и локальные веса; на исходной машине данные уже лежат в dataset существующего репозитория.
`requirements.lock.txt` фиксирует фактически использованное окружение macOS/Python 3.12;
для другой платформы используйте совместимые диапазоны в `requirements.txt`.

В ZIP находятся локальные веса E5. Если папки `outputs/models/multilingual-e5-small` нет,
однократно выполните `python scripts/download_model.py`. Это только загрузка открытых весов;
дальнейшее обучение/инференс выполняется локально, без внешнего API и с offline-флагом.
Revision записан в `outputs/models/dense_manifest.json`.

Выбранный ноутбук можно исполнить отдельно:

```bash
python scripts/run_notebooks.py 02_baselines.ipynb
```

Зависимости артефактов: `00 → 01 → 02 → 03`; `04` требует только `01` и локальную модель;
`05` требует `02–04`, `06` требует `05`. Для последовательного запуска достаточно команды выше.
В ноутбуках используются разные префиксы (`eda_`, `fe_`, `bl_`, `hy_`, `de_`, `fu_`, `su_`),
поэтому один notebook не перезаписывает исследовательские переменные другого в общем kernel.
Тяжёлые объекты всё равно занимают память: отдельные kernels предпочтительнее для полного прогона.

## Ноутбуки

| Ноутбук | Содержание |
|---|---|
| `00_EDA` | schema, null/empty, cardinality, повторы, дрейф, overlap, конфликты item metadata, хвосты, boxplots, корреляции, PCA |
| `01_feature_engineering` | тексты/числа, признаки со смыслом, warm/cold и независимые rank_train/dev/test, assertions утечек |
| `02_baselines` | популярность, overlap, word/char TF-IDF, BM25, title BM25, NB→microcategory, exact/context/near-query history |
| `03_hybrid_retrieval` | RRF, мягкая география, union ceiling, небольшая dev-сетка, ablations |
| `04_dense_retrieval` | локальный E5, exact cosine, возобновляемый кэш, mining неразмеченных hard negatives, residual query adapter |
| `05_supervised_fusion` | естественные отложенные кандидаты, CatBoost, выбор по dev, importance и permutation ablations |
| `06_submission` | один holdout report, OOF refit, ограничение корпуса до top-k, answer.csv, строгий validator |

## Данные и честность оценки

- 497 673 положительные строки, 344 825 уникальных train items.
- 189 212 benchmark items, 2 452 benchmark queries.
- 30 619 полных дублей; 30 654 повторов `(полный контекст, item)` после нормализации.
- 37.48% benchmark текстов встречаются в train после нормализации, но лишь 4.45% полных контекстов.
- Только 9.59% benchmark items встречаются в train. Доступная история не заменяет content retrieval.
- У 16.90% позитивов отличаются location ID. Поэтому нет безусловного отсечения по локации.

Локальный каталог — 515 895 уникальных объявлений из объединения файлов, без добавления позитивов
в выдачу. При конфликте metadata приоритет у benchmark; EDA отдельно проверяет фактические конфликты.
Финальный корпус — строго 189 212 benchmark items.

Cold: все контексты одного нормализованного текста целиком отложены.
Warm: отложен полный контекст знакомого текста; другие контексты этого текста разрешены в base.
Тексты rank_train/dev/test между собой не пересекаются. Один текст — один оцениваемый контекст.
Повторы полной пары запрещены между частями. Warm дополнительно разбит на новые/знакомые пары и cold items.
Исторический выбор item в другом контексте — доступный сигнал, но может делать warm score оптимистичнее.

Feature engineering не выдумывает CTR, user embeddings, timestamps или координаты запроса.
Дистанция — приближение до центра location из каталога, а не расстояние до пользователя.
Рейтинг сглаживается на 20 условных отзывов к медиане каталога; цену сравниваем внутри microcategory.
PCA и корреляции — описательная диагностика, не feature selection по константному positive-only target.

Fusion обучается на кандидатов `rank_train`, полученных кандгенами, которые не видели их полный контекст.
Dev выбирает конфигурацию; test оценивается после фиксации. Затем разрешён refit для сдачи.
Кандидаты без разметки — шумные отрицательные, не подтверждённые нерелевантные объявления.
Отсутствующие позитивы не вставляются в candidate pool. Потолок recall учитывает все eval-запросы.

Официальной test-разметки нет. Локальные числа не являются leaderboard score и не обещают его.
Состав каталогов и отбор запросов различаются, timestamps отсутствуют: это не временная оценка.

## Артефакты

- `outputs/figures/`: реальные графики.
- `outputs/validation/`: таблицы метрик, аудит, timings, ablations, ошибки и hard negatives.
- `outputs/models/`: выбранная конфигурация, CatBoost, query adapter, локальный E5.
- `outputs/cache/`: восстановимые индексы, кандидаты, embeddings и разбиения. Не включаются в переносимый ZIP.
- `outputs/submissions/answer.csv`: сдача; `validation.json`: результат проверки формата.
- `reports/research.md`: ссылки, сравнение близких решений, ограничения переноса.
- `reports/eda_findings.md`: выводы фактического EDA.

Кэш нужно перестроить после изменения данных, нормализации или параметров индекса.
`00` пишет SHA-256 исходных файлов. Dense checkpoint дополнительно проверяет signature и порядок items.
Последовательный полный запуск пересоздаёт индексы и кандидатов; не считайте вручную перемешанные кэши совместимыми.

## Проверки

```bash
python -m unittest discover -s tests -v
python scripts/validate_answer.py
```

Проверяются macro-усреднение и отсутствующие выдачи, формула BM25, ties/local top-k, нормализация,
RRF, ведущие нули ID и ошибочные варианты CSV. В notebook `01` есть проверки разбиений.
CSV — UTF-8, ровно `query_id,answer`, один пробел между item IDs, не более 50 уникальных ID из корпуса.
Пустая выдача разрешена контрактом, но финальная генерация заполняет до 50 детерминированным fallback.

Использованы NumPy, pandas, SciPy, scikit-learn, CatBoost, PyTorch, Sentence Transformers и открытая модель E5.
Чужой код решений не копировался. Git history не менялась, commit/push и отправка на платформу не выполнялись.
