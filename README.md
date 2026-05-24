# AI-модуль выявления и приоритизации банковских риск-инфоповодов

Учебный проект по разработке ML/NLP-сервиса.

Цель проекта — разработать MVP AI-модуля, который анализирует новостной поток, находит упоминания банков, сопоставляет их с единым справочником организаций, определяет тип риск-инфоповода и рассчитывает explainable risk score для приоритизации карточек в дашборде.

## Домашние задания

- [Домашнее задание №1. Бизнес-анализ](homework_01_business_understanding/README.md)
- [Домашнее задание №2. Продуктовый прототип](homework_02_product_prototype/README.md)
- [Домашнее задание №3. Benchmark](homework_03_benchmark/README.md)
- [Домашнее задание №4. Датасет для решения задачи](homework_04_dataset/README.md)

## Быстрый запуск блока ДЗ №4

```bash
pip install -r requirements.txt

python -m src.data.build_candidates --max_candidates 5000 --max_scanned_rows 300000 --seed 42
python -m src.data.make_annotation_dataset --mode natural --annotation_rows 500 --seed 42
python -m src.data.make_annotation_dataset --mode risk_enriched --annotation_rows 300 --exclude_existing data/processed/annotation_template.csv --output data/processed/annotation_template_risk_enriched.csv --seed 43
python -m src.data.assist_manual_annotation
```

`natural_sample` нужен для оценки реальной структуры банковских упоминаний. `risk_enriched_sample` нужен для донабора положительных риск-примеров. Это нормально: direct-risk события редки в естественном новостном потоке.

Автоматическая разметка является только предразметкой. Финальная разметка считается готовой только после ручной проверки человеком и заполнения финальных полей.

После ручной разметки `data/processed/annotation_template.csv` или assisted-файла с переносом проверенных значений в финальные поля:

```bash
python -m src.data.finalize_labeled_dataset
python -m src.reports.build_eda_outputs
```
