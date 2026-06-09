# Домашнее задание №5-6. ML-моделирование и анализ качества модели

## 1. Цель работы

Построить и оценить ML-пайплайн для мониторинга банковских риск-сигналов по новостным фрагментам.

Основная задача — binary classification:

```text
title + text_fragment + entity_norm -> alert_flag
```

Модель должна определить, содержит ли новостной фрагмент риск-сигнал, который потенциально нужно показать аналитику.

## 2. Данные

Датасет подготовлен на этапе ДЗ №4 на базе русскоязычного новостного корпуса RuNews / `IlyaGusev/ru_news`.

Используется файл:

```text
homework_04_dataset/data/dataset_for_training.csv
```

Единица наблюдения — фрагмент новости вокруг упоминания банка.

Основной target:

```text
alert_flag
```

- `0` — риск-сигнал не выявлен;
- `1` — фрагмент содержит риск-сигнал.

Поле `risk_type_4cls` рассматривается как дополнительная разметка для будущего расширения пайплайна, но в текущей clean-версии multi-class модель не входит в основной modeling pipeline.

## 3. Контроль leakage

Используется готовый `train/valid/test split`, сформированный на этапе подготовки датасета. Новое random split не выполняется.

Не используются как features:

- `sample_id`;
- `event_group_id`;
- `split`;
- `alert_flag`;
- `risk_type_4cls`;
- `found_risk_keywords`.

`event_group_id` используется для контроля утечек между split, но не подаётся в модель как признак.

`found_risk_keywords` исключён из признаков, потому что связан с правиловой логикой поиска риск-слов и может привести к leakage.

## 4. Модели

В clean-версии проверяются:

1. `TF-IDF word`
2. `TF-IDF char`
3. `Combined TF-IDF`
4. `Tuned combined TF-IDF`
5. `LLM-distilled TF-IDF (teacher: GPT-4.1)`
6. `LLM-distilled TF-IDF (teacher: GPT-5.5)`

Логика экспериментов:

```text
простые TF-IDF baseline
-> combined word+char TF-IDF
-> tuning LogisticRegression
-> LLM distillation
```

## 5. LLM distillation

LLM distillation используется как advanced-подход.

Большая языковая модель выступает как teacher и возвращает:

- binary label;
- confidence.

Student-модель — компактная `Combined TF-IDF + LogisticRegression`.

Teacher labels формируются только для train-выборки и кэшируются в:

```text
homework_05_06_modeling/reports/llm_distillation/
```

Valid используется для выбора teacher-модели и параметров distillation. Test не используется для выбора модели или параметров.

Для GPT-5.5 использовался Responses API, structured JSON output и увеличенный output budget, чтобы получить валидные teacher labels.

## 6. Метрики

Для binary classification рассчитываются:

- Precision;
- Recall;
- F1;
- ROC-AUC;
- PR-AUC;
- confusion matrix.

Основная метрика выбора модели — `F1` на valid, потому что она балансирует Precision и Recall.

Test используется только для финальной проверки и диагностического сравнения, а не для выбора модели.

## 7. Основные файлы

Основной clean notebook:

```text
homework_05_06_modeling/notebooks/05_06_modeling_and_metrics_clean.ipynb
```

Краткий summary-отчёт:

```text
homework_05_06_modeling/reports/modeling_summary.md
```

Optional LLM benchmark:

```text
homework_05_06_modeling/notebooks/optional_llm_benchmark.ipynb
```

Датасет:

```text
homework_04_dataset/data/dataset_for_training.csv
```

LLM distillation cache:

```text
homework_05_06_modeling/reports/llm_distillation/
```

## 8. Как запустить

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить notebook:

```bash
jupyter notebook homework_05_06_modeling/notebooks/05_06_modeling_and_metrics_clean.ipynb
```

Если LLM teacher labels уже сохранены в cache, повторные API-вызовы не требуются.

## 9. Ограничения

- датасет небольшой;
- положительный класс встречается реже отрицательного;
- метрики Precision, Recall и F1 чувствительны к нескольким объектам;
- качество distillation зависит не только от мощности teacher-модели, но и от prompt, confidence calibration и качества исходной разметки;
- результат GPT-5.5 на test не используется для переизбрания модели задним числом.

## 10. Что улучшать дальше

- расширить размеченную выборку;
- вручную проверить false positives и false negatives;
- уточнить `event_group_id` для более строгого контроля near-duplicates;
- проверить стабильность результатов на новом holdout;
- подобрать threshold под реальную нагрузку аналитиков;
- отдельно развить multi-class классификацию `risk_type_4cls` как следующий этап.
