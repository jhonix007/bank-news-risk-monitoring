# Homework 07: LLM Distillation Postprocessing

Это ДЗ №7 по ML Engineering. Работа анализирует LLM-distillation pipeline из ДЗ №5-6:

```text
teacher prompt -> LLM teacher labels on train -> soft_y -> student model -> valid/test predictions
```

`Teacher prompt v1` взят из ДЗ №5-6. По ошибкам `student v1` на valid был сформирован refined teacher prompt `v2b`; на основе teacher labels `v2b` обучен `student v2b`. Postprocessing применяется к probabilities student-модели, а не к прямым LLM predictions.

LLM используется только как teacher для train-разметки. На `valid` и `test` предсказывает compact student model `Combined word+char TF-IDF + LogisticRegression`.

Цель ДЗ №7:

- взять teacher prompt v1 из ДЗ №5-6;
- получить teacher labels v1 на train и обучить student v1 через soft labels;
- проанализировать ошибки student v1 на valid;
- сформировать refined teacher prompt v2b на основе valid error analysis;
- получить teacher labels v2b на train и обучить student v2b;
- сравнить student v1 и student v2b на valid;
- выполнить threshold tuning и review-zone для финальной student-модели;
- выбрать финальную схему только по valid;
- один раз проверить выбранную схему на test;
- подробно разобрать метрики и ошибки.

Основной notebook:

```text
homework_07_postprocessing/notebooks/07_postprocessing_and_error_analysis.ipynb
```

Входной датасет:

```text
homework_04_dataset/data/dataset_for_training.csv
```

Teacher label cache:

```text
homework_07_postprocessing/reports/teacher_labels/train_teacher_prompt_v1.csv
homework_07_postprocessing/reports/teacher_labels/train_teacher_prompt_v2b.csv
```

Student predictions:

```text
homework_07_postprocessing/reports/student_predictions/valid_student_v1.csv
homework_07_postprocessing/reports/student_predictions/valid_student_v2b.csv
homework_07_postprocessing/reports/student_predictions/test_student_final.csv
```

## Итог эксперимента

Refined teacher prompt v2b дал небольшой прирост valid F1 относительно student v1 и повысил Recall, ROC-AUC и PR-AUC. Основной эффект - снижение числа false negatives, то есть модель стала лучше находить риск-сигналы.

При этом Precision снизился, а число false positives выросло. Поэтому результат prompt refinement интерпретируется как recall-oriented trade-off, а не как универсальное улучшение всех метрик.

Threshold tuning был выполнен на valid. Лучшим full-coverage threshold оказался 0.50, поэтому дополнительное ужесточение порога не применялось. Test использовался только для финальной проверки выбранной схемы.

## API и cache

API calls выключены по умолчанию:

```python
RUN_LLM_API = False
```

Если teacher cache существует, notebook использует его. Если cache отсутствует, notebook выводит инструкцию: нужно либо положить cache-файлы, либо явно включить `RUN_LLM_API = True`.

## Valid/test protocol

`valid` используется для анализа ошибок, доработки teacher prompt и выбора postprocessing. `test` используется только для финальной проверки. Prompt, threshold и review-zone нельзя дорабатывать после просмотра test.
