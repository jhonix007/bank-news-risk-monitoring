# ДЗ №4. Датасет для решения задачи

## 1. Источник и состав данных

Источник данных — Hugging Face dataset `IlyaGusev/ru_news`. Датасет читается потоково через `datasets streaming=True`; полный RuNews не скачивается в проект и не коммитится в Git.

Используются поля `title`, `text`, `timestamp`, `url`, `source`. Единица наблюдения — один фрагмент новости вокруг одного найденного банка.

## 2. Алгоритм формирования выборки

Выборка формировалась из потока новостей: сначала выполнялся поиск банков по словарю алиасов, затем неоднозначные алиасы фильтровались по банковскому контексту. Для каждой найденной пары `новость + банк` выделялся фрагмент вокруг упоминания, после чего удалялись дубли по `url + entity_norm + text_fragment`.

Финальный датасет собран из основной выборки банковских упоминаний и дополнительного добора риск-кандидатов. Колонка `dataset_part` используется как технический признак происхождения строки и помогает контролировать качество выборки, но не является целевой переменной для обучения.

Все финальные метки проверялись вручную. Автоматическая разметка использовалась только как предразметка.

## 3. Состав финального датасета

Финальный датасет:

```text
data/processed/news_risk_dataset_labeled.csv
```

Всего строк: `835`.

Датасет включает основную выборку банковских упоминаний и дополнительный добор риск-кандидатов. Техническая колонка `dataset_part` оставлена для контроля происхождения строк.

Финальные колонки:

```text
sample_id
dataset_part
source
url
published_at
published_year
published_month
title
text_fragment
entity_mention
entity_norm
risk_type_candidate
found_risk_keywords
risk_type
entity_relevance
alert_flag
alert_reason
label_quality
risk_score_v1
split
review_status
review_comment
```

Suggested-поля не входят в финальный датасет: в нем оставлены только проверенные поля разметки и аналитические поля.

## 4. Схема разметки

- `entity_norm` — нормализованное название найденного банка.
- `entity_relevance` — связь риска с найденным банком: `direct`, `indirect`, `mentioned_only`, `unclear`.
- `risk_type` — финальный тип риска.
- `alert_flag` — нужно ли показывать карточку специалисту.
- `alert_reason` — короткое объяснение причины алерта.
- `label_quality` — качество разметки: `ok`, `ambiguous`, `need_review`.
- `risk_score_v1` — объяснимый скор риска от 0 до 100.

`risk_score_v1` не размечается вручную, а рассчитывается по формуле: вес типа риска + вес релевантности банка + вес алерта, максимум 100.

## 5. Базовый EDA и выводы для моделирования

Распределение `risk_type`:

- `no_risk`: 729
- `sanctions`: 60
- `fraud_phishing`: 15
- `legal_regulatory`: 12
- `operational_issue`: 9
- `data_leak_security`: 7
- `other_risk`: 3

Распределение `alert_flag`:

- `0`: 757
- `1`: 78

Основные банки: ВТБ, Сбербанк, Газпромбанк, Альфа-Банк, Россельхозбанк. Основные источники: `buriy`, `ods_tass`, `lenta`, `taiga_fontanka`, `telegram_contest`. Диапазон годов: 2003-2020.

Выводы: большинство строк относятся к `no_risk`, а `alert_flag = 1` встречается редко. Это подтверждает наличие информационного шума в банковских упоминаниях. Для обучения бинарной модели нужен balanced train. `risk_type` можно использовать как multi-class задачу, но качество по редким классам будет ограничено малым числом примеров.

Сильный дисбаланс `risk_type` является ожидаемым: `no_risk` доминирует, потому что большинство банковских упоминаний не является риск-инфоповодом. Это не делает датасет непригодным: для текущего этапа основной задачей первой модели является бинарная классификация `alert_flag`, то есть определение, нужно ли показывать карточку специалисту.

Для обучения `alert_flag` создается отдельный balanced train, при этом `valid` и `test` остаются ближе к исходному распределению. `risk_type` сохраняется как дополнительная разметка для анализа ошибок и будущего расширения проекта. Полноценное обучение multi-class модели по `risk_type` возможно в следующей итерации после накопления большего числа примеров по редким классам.

Основные EDA-графики для отчета:

- `homework_04_dataset/eda_outputs/alert_flag_distribution.png`
- `homework_04_dataset/eda_outputs/risk_type_distribution_full.png`
- `homework_04_dataset/eda_outputs/risk_type_distribution_without_no_risk.png`
- `homework_04_dataset/eda_outputs/entity_relevance_distribution.png`
- `homework_04_dataset/eda_outputs/text_fragment_length_hist.png`
- `homework_04_dataset/eda_outputs/risk_keyword_distribution_top30.png`

Дополнительные технические проверки:

- `homework_04_dataset/eda_outputs/dataset_part_distribution.png`
- `homework_04_dataset/eda_outputs/dataset_part_vs_alert_flag.png`
- `homework_04_dataset/eda_outputs/risk_type_distribution_log_scale.png`

### Текстовые характеристики

Дополнительно анализируются:

- длина заголовков;
- длина текстовых фрагментов;
- количество слов;
- частота риск-слов;
- топ слов для риск- и нериск-фрагментов.

Текстовый EDA нужен, чтобы проверить пригодность фрагментов для baseline-моделей `TF-IDF + Logistic Regression` / `LinearSVC` и понять, какие слова чаще всего формируют риск-сигналы.

## 6. Качество разметки

Сначала использовалась rule-based / assisted предразметка. Suggested-поля не входят в финальный датасет. Финальные поля проверялись вручную.

Основные ошибки предразметки:

- банк был кредитором, комментатором или участником общего контекста;
- риск относился не к банку;
- ключевые слова вроде "иск", "санкции", "сбой" давали ложные срабатывания.

Для улучшения качества был добавлен targeted-добор редких классов. Спорные строки отмечались через `label_quality = ambiguous / need_review`.

## 7. Стратегия валидации

Используется разбиение `train / valid / test = 70 / 15 / 15`. При разбиении учитывается группировка по `url`: одна и та же новость не должна попадать одновременно в обучение и проверку. Это снижает риск утечки информации между train и test.

Финальный датасет не балансировался искусственно, потому что он должен отражать реальную структуру банковского новостного потока: большинство упоминаний банка не являются прямыми риск-инфоповодами. Это важно для проверки способности модели отделять риск-карточки от информационного шума.

В результате подготовки данных создаются четыре modeling-файла:

```text
data/processed/modeling/news_risk_full_train.csv
data/processed/modeling/news_risk_full_valid.csv
data/processed/modeling/news_risk_full_test.csv
data/processed/modeling/news_risk_train_balanced_alert.csv
```

`news_risk_full_train.csv`, `news_risk_full_valid.csv` и `news_risk_full_test.csv` сохраняют исходное распределение классов. `valid` и `test` не балансируются искусственно, чтобы оценка качества была ближе к реальному продуктовому сценарию, где риск-инфоповоды встречаются редко.

Отдельный файл `news_risk_train_balanced_alert.csv` используется только для обучения первой бинарной baseline-модели по `alert_flag`. Он строится только на train-части: берутся все положительные примеры `alert_flag = 1` и случайная подвыборка отрицательных примеров `alert_flag = 0` с коэффициентом `negative_ratio = 2`.

Таким образом:

- полный датасет и EDA показывают реальную структуру данных;
- `valid` и `test` остаются честными и несбалансированными;
- балансировка применяется только к обучающей части, чтобы модель не выучила тривиальное правило «почти всегда ставить `alert_flag = 0`».

Для `risk_type` полноценная оценка по редким классам ограничена: классы `operational_issue`, `data_leak_security` и `other_risk` представлены малым числом примеров. Поэтому `risk_type` используется как дополнительная метка и направление для следующей итерации расширения датасета.

## 8. Как планируется обучать модель

Основная baseline-задача первой модели:

```text
text_fragment + entity_norm -> alert_flag
```

То есть модель должна определить, нужно ли показывать фрагмент новости специалисту как риск-карточку. Это напрямую связано с продуктовой задачей: сократить ручной просмотр новостного потока и оставить только приоритетные сообщения.

Первая модель: `TF-IDF + Logistic Regression`.

### Использование balanced train

Финальный датасет `data/processed/news_risk_dataset_labeled.csv` не балансируется искусственно. Он остаётся полным и используется для EDA, анализа качества разметки и формирования честных `valid` / `test`.

Для обучения бинарной модели по `alert_flag` используется отдельный файл:

```text
data/processed/modeling/news_risk_train_balanced_alert.csv
```

Он формируется только из train-части. В него попадают:

- все положительные train-примеры `alert_flag = 1`;
- случайная подвыборка отрицательных train-примеров `alert_flag = 0`;
- соотношение отрицательных к положительным задаётся параметром `negative_ratio = 2`.

В текущей версии подготовлены следующие modeling-файлы:

- `news_risk_full_train.csv`;
- `news_risk_full_valid.csv`;
- `news_risk_full_test.csv`;
- `news_risk_train_balanced_alert.csv`.

Такой подход позволяет обучать модель на более информативном train-наборе, но оценивать её на несбалансированных `valid` и `test`, которые ближе к реальному потоку.

Дополнительно планируется сравнить:

- rule-based baseline;
- `TF-IDF + Logistic Regression`;
- `TF-IDF + LinearSVC`;
- `risk_type` как вспомогательную multi-class задачу.

`risk_score_v1` не предсказывается напрямую моделью. Он рассчитывается по объяснимой формуле на основе `risk_type`, `entity_relevance` и `alert_flag`.

Метрики:

- для `alert_flag`: Precision, Recall, F1, PR-AUC, confusion matrix;
- для top-K карточек: Precision@K;
- для `risk_type`: macro F1 только как ориентир, с оговоркой о редких классах.

## 9. Ограничения

- Датасет учебный.
- RuNews не равен промышленному медиамониторингу.
- Классы несбалансированы.
- Редкие классы недопредставлены.
- Разметка частично субъективна.
- Нет валидации вторым разметчиком.

## 10. Состав файлов для сдачи

- `data/processed/news_risk_dataset_labeled.csv`
- `data/processed/modeling/`
- `notebooks/04_data_understanding_eda.ipynb`
- `homework_04_dataset/annotation_guideline.md`
- `homework_04_dataset/eda_outputs/`
- `src/data/`
- `src/features/`
- `src/reports/`

## Команды

Построение EDA:

```bash
python -m src.reports.build_eda_outputs --input data/processed/news_risk_dataset_labeled.csv --output-dir homework_04_dataset/eda_outputs
```

Подготовка файлов для обучения:

```bash
python -m src.data.build_training_splits --input data/processed/news_risk_dataset_labeled.csv --output-dir data/processed/modeling --negative-ratio 2 --seed 42
```
