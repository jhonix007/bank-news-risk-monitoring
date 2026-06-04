# ДЗ №4. Датасет для ML-задачи мониторинга банковских рисков

## Цель

Подготовить датасет для обучения модели, которая выделяет из новостного потока банковские риск-сигналы.

Основная ML-задача:

```text
title + text_fragment + entity_norm -> alert_flag
```

`alert_flag = 1` означает, что фрагмент стоит показать специалисту как риск-сигнал. `alert_flag = 0` означает, что банковское упоминание не требует риск-карточки.

## Источник данных

Исходная база — RuNews / `IlyaGusev/ru_news`. Использовались русскоязычные новости. Единица наблюдения — текстовый фрагмент вокруг упоминания банка. Одна новость может дать несколько строк, если в ней упоминаются разные банки.

## Финальный датасет

Основной файл:

```text
homework_04_dataset/data/dataset_for_training.csv
```

Размер: 839 строк.

Ключевые колонки:

- `sample_id` — технический ID строки;
- `event_group_id` — группа одного или близкого инфоповода;
- `split` — train/valid/test;
- `title`, `text_fragment`, `entity_norm` — основные признаки для baseline;
- `source`, `published_year`, `published_month` — контекстные признаки, использовать осторожно;
- `alert_flag` — основной target;
- `risk_type_4cls` — дополнительная 4-классовая разметка.

Классы `risk_type_4cls`:

- `no_risk`;
- `cyber_risk`;
- `operational_risk`;
- `regulatory_risk`.

## Основной EDA

Главный аналитический отчёт находится в ноутбуке:

```text
homework_04_dataset/notebooks/04_dataset_eda_and_validation.ipynb
```

В ноутбуке собраны код, таблицы, графики и текстовые выводы по каждому ключевому блоку: структура датасета, пропуски, дубликаты, распределения target, текстовые характеристики, источники, годы, качество разметки, split и leakage-анализ.

## Краткие EDA-выводы

Распределение `alert_flag`:

- `0`: 585;
- `1`: 254.

Положительный класс встречается реже отрицательного, поэтому accuracy не должна быть основной метрикой. Для оценки модели нужны precision, recall, F1 и PR-AUC.

Распределение `risk_type_4cls`:

- `no_risk`: 585;
- `cyber_risk`: 111;
- `operational_risk`: 97;
- `regulatory_risk`: 46.

Датасет подходит для бинарного baseline. Multi-class задача возможна как дополнительный эксперимент, но классы риска меньше, поэтому метрики по ним нужно интерпретировать осторожно.

## Качество разметки

Разметка проверяется логическими правилами:

- `risk_type_4cls = no_risk` должен соответствовать `alert_flag = 0`;
- риск-классы должны соответствовать `alert_flag = 1`;
- риск должен относиться к найденному банку, а не просто встречаться рядом с ним.

Подробные проверки приведены в notebook и кратком отчёте:

```text
homework_04_dataset/reports/dataset_quality_summary.md
```

## Train/valid/test split

Split построен как group-aware split по `event_group_id`. Это нужно, чтобы один и тот же или близкий инфоповод не попадал одновременно в train и valid/test.

Распределение:

- `train`: 587;
- `valid`: 126;
- `test`: 126.

Valid/test не балансируются искусственно, чтобы оценка была ближе к реальному распределению новостного потока.

## Exact duplicates и near-duplicates

В notebook проверяются:

- точные дубли по `sample_id`;
- точные дубли по `text_fragment`;
- точные дубли по `title + text_fragment + entity_norm`;
- near-duplicates между split через TF-IDF cosine similarity.

`event_group_id` используется для снижения риска leakage. Если near-duplicates всё ещё находятся между split, это фиксируется как ограничение и рекомендация для следующей итерации: строить более строгий `event_cluster_id`.

## Leakage-анализ

Для первой честной binary-модели не использовать как признаки:

- `sample_id`;
- `event_group_id`;
- `split`;
- `alert_flag`;
- `risk_type_4cls`;
- `found_risk_keywords`.

Рекомендуемые признаки для baseline:

```text
title + text_fragment + entity_norm
```

`source`, `published_year`, `published_month` можно проверять отдельно, но есть риск source bias и temporal drift.

## Рекомендуемый baseline

Первая модель:

```text
TF-IDF + Logistic Regression
```

или:

```text
TF-IDF + LinearSVC
```

Для дисбаланса использовать `class_weight='balanced'`. Основные метрики: precision, recall, F1, PR-AUC, confusion matrix. Для продуктового сценария дополнительно полезен Precision@K.

Вторая возможная задача — многоклассовая классификация:

```text
title + text_fragment + entity_norm -> risk_type_4cls
```

Её можно обучать как дополнительный эксперимент, чтобы модель не только находила риск-сигнал, но и относила его к одному из классов: `no_risk`, `cyber_risk`, `operational_risk`, `regulatory_risk`. Для этой задачи основная метрика — macro F1, потому что классы риска меньше и важна устойчивость качества по каждому классу.

Важно: для бинарной модели `risk_type_4cls` нельзя использовать как feature, но для отдельной multi-class модели это допустимый target.

## Ограничения

- Датасет учебный и небольшой.
- Источник — исторический RuNews, а не промышленный медиамониторинг.
- Есть временной сдвиг: новости покрывают 2002-2020 годы.
- Источников немного, поэтому возможен source bias.
- Near-duplicate grouping снижает риск leakage, но не гарантирует идеальное объединение всех семантически близких инфоповодов.
- Для промышленной версии нужна проверка вторым разметчиком и свежий holdout.

## Состав файлов

```text
homework_04_dataset/README.md
homework_04_dataset/annotation_guideline.md
homework_04_dataset/notebooks/04_dataset_eda_and_validation.ipynb
homework_04_dataset/data/dataset_for_training.csv
homework_04_dataset/reports/dataset_quality_summary.md
```
