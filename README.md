# HW7 · Time-series features + JSONB event-log у PostgreSQL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qprxAuC7uMkPAsYd0d-SygUTQzJY_Bky?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
[![Dataset](https://img.shields.io/badge/Dataset-NYC%20TLC%20Yellow%20Taxi-20BEFF)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Відтворюваний pipeline для двох типів даних, з яких зазвичай будують ML-ознаки:

1. **Time-series:** реальні поїздки NYC Yellow Taxi → щоденний агрегат по зонах → `lag` / `rolling` / `expanding` ознаки через window functions.
2. **Напівструктурований event-log:** синтетичні події в `JSONB` → `@>`, `?`, `?|`, JSONPath → GIN та expression B-tree індекси з виміром ефекту через `EXPLAIN (ANALYZE, BUFFERS)` → порівняння з нормалізованою таблицею.


## Зміст

- [Структура репозиторію](#структура-репозиторію)
- [Швидкий старт](#швидкий-старт)
- [Дані](#дані)
- [Pipeline](#pipeline)
- [Ключові результати](#ключові-результати)
- [Відтворюваність](#відтворюваність)

## Структура репозиторію

```text
.
├── 07_time_series_features.ipynb   # notebook з усіма output-ами + Reflection
├── requirements.txt                # Python-залежності
└── README.md
```

> Parquet-файли не зберігаються в репозиторії: notebook сам завантажує їх з офіційного джерела TLC.

## Швидкий старт

### Google Colab (рекомендовано)

1. Відкрийте notebook кнопкою **Open in Colab** вище.
2. Виберіть **Runtime → Restart session and run all**.

Перша клітинка встановить і запустить PostgreSQL, задасть пароль `postgres` і встановить Python-залежності. Жодних ручних дій не потрібно. Повний прогін займає кілька хвилин, з них ~2 хв іде на завантаження та `COPY` ~6.5 млн рядків.

### Локально

Потрібен PostgreSQL 14+ на `localhost:5432` з користувачем `postgres` / паролем `postgres`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook 07_time_series_features.ipynb
```

> [!NOTE]
> Перша клітинка notebook містить Colab-специфічні команди (`apt-get`, `service postgresql start`). Локально її можна пропустити, якщо PostgreSQL уже запущено. Notebook очікує, що Parquet-файли лежатимуть у `/content/`; локально змініть шлях `parquet_path` у клітинці завантаження.

## Дані

| Набір | Джерело | Обсяг |
|---|---|---|
| NYC TLC Yellow Taxi, `2026-07` і `2026-08` | [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet` | 6 866 825 raw → **6 505 128** після очищення (~104.9k поїздок/день) |
| Синтетичний event-log | `generate_series` у PostgreSQL (детерміновано) | 50 000 подій, 5 типів |

**Чому два повні місяці без семплування.** З одним місяцем `trips_lag_30d` був би заповнений лише для одного дня. Семпл (`head()` / `sample()`) спотворює `trip_count` і створює фальшиві «нульові» дні. З повними даними нуль означає, що в зоні справді не було поїздок.

**Очищення.** Правила застосовуються в pandas перед `COPY`, а частину з них дублює `WHERE` у SQL-агрегації. Notebook друкує звіт про кількість рядків, відкинутих кожним правилом.

| Правило | 2026-07 | 2026-08 |
|---|---:|---:|
| pickup поза місяцем партиції | 46 | 15 |
| `fare_amount` ∉ (0, 500) | 17 031 | 17 360 |
| `trip_distance` ∉ (0, 100) | 128 494 | 119 194 |
| `PULocationID` IS NULL | 0 | 0 |
| тривалість ∉ (0, 180] хв | 43 318 | 44 058 |
| середня швидкість > 80 mph | 42 212 | 40 998 |
| **Залишилось** | **3 343 698** | **3 161 430** |

> [!IMPORTANT]
> **Stub-schema fallback.** Якщо TLC-файли недоступні (немає мережі), notebook автоматично (`DATA_SOURCE = "stub"`) створює `t7_hw_taxi_daily` зі stub-DDL із завдання: 60 днів × 5 зон = 300 рядків, фіксована дата замість `CURRENT_DATE`. Збережені output-и в цьому репозиторії отримано на **реальних TLC-даних** (`DATA_SOURCE = tlc`).

## Pipeline

### 1. Setup та завантаження

- Запуск PostgreSQL 16 у Colab, перевірка `version()`.
- `nyc_taxi_raw`: таблиця з явною схемою. Завантаження через `COPY ... FROM STDIN`, бо `to_sql` для мільйонів рядків занадто повільний.
- `t7_hw_taxi_daily (pickup_zone_id, ts_day, trip_count, avg_fare, total_distance)`: PK `(pickup_zone_id, ts_day)`, `CHECK`-обмеження. Таблиця має **16 182 рядки** (261 зона × 62 дні).
  - `ts_day DATE` — **локальний NYC-день**, бо raw timestamps у TLC є NYC wall-clock часом.
  - **Щільний календар:** дні без поїздок заповнюються `trip_count = 0`, `avg_fare = NULL`. Завдяки цьому `LAG(…, k)` зсуває саме на *k днів*, а не на *k рядків*.
- `t7_hw_taxi_trips_typed`: demo production-зберігання моменту події як `TIMESTAMPTZ` через `AT TIME ZONE 'America/New_York'`.

### 2. Time-series ML features — `v_t7_hw_taxi_features`

Усі вікна містять `PARTITION BY pickup_zone_id`, щоб `LAG` не «перетікав» між зонами.

| Ознака | Функція | Window frame |
|---|---|---|
| `trips_lag_1d`, `trips_lag_7d`, `trips_lag_30d` | `LAG(trip_count, k)` | `ORDER BY ts_day` |
| `rolling_avg_fare_7d` | `AVG(avg_fare)` | `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW` |
| `rolling_std_fare_30d` | `STDDEV_SAMP(avg_fare)` | `RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW` |
| `cumulative_trips`, `expanding_avg_trips` | `SUM` / `AVG(trip_count)` | `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` |
| `target_trips_next_1d` | `LEAD(trip_count, 1)` | ML-target `t+1`, без leakage |

Додатково:

- демо `ROWS` vs `RANGE` на ряді з пропущеними днями (off-by-one);
- assertion-перевірка на **8 352** рядках view: `lag1/lag7/lag30/cumsum_mismatch = 0`.

### 3. JSONB event-log — `t7_hw_events`

| `event_type` | Ключі `payload` |
|---|---|
| `page_view` | `url`, `referrer`, `device` |
| `click` | `element_id`, `page_url`, `device` |
| `add_to_cart` | `product_id`, `price`, `quantity`, `tags[]` |
| `checkout` | `order_id`, `total`, `items_count`, `payment_method` |
| `recommendation_shown` | `model_name`, `model_version`, `recommendations[]`, `experiment` |

Запити: containment `@>`, key existence `?` / `?|`, JSONPath (`jsonb_path_query`, `jsonb_path_exists`).

### 4. Індекси та `EXPLAIN (ANALYZE, BUFFERS)`

GIN `jsonb_path_ops`, GIN `jsonb_ops` і expression B-tree на `payload ->> 'device'`. Для кожного запиту збережено план до і після індексу, виміряно вартість запису та зібрано зведену таблицю планів.

### 5. JSONB vs normalized

`t7_hw_checkouts_norm` — typed-таблиця для `checkout` з `CHECK`-обмеженнями. Один і той самий агрегат (`AVG(total)` по `payment_method`) порівнюється у двох версіях. Як bonus додано еквівалент на MongoDB aggregation pipeline (pseudocode).

## Ключові результати

Виміряно в Colab, PostgreSQL 16.15, 50 000 подій (таблиця 8000 kB).

| Запит | Індекс | План | Total cost | Час |
|---|---|---|---:|---:|
| `@> {"payment_method":"card"}` | — | Seq Scan | 1756.60 | 34.0 ms |
| `@> {"payment_method":"card"}` | GIN `jsonb_path_ops` (1144 kB) | Bitmap Index → Heap Scan | 1091.69 | **5.4 ms** |
| `payload ->> 'device' = 'mobile'` | лише GIN | Seq Scan | 1750.63 | 9.5 ms |
| `payload ->> 'device' = 'mobile'` | expression B-tree (360 kB) | Bitmap Index → Heap Scan | 1242.94 | **5.3 ms** |
| `payload ? 'tags'` | лише `jsonb_path_ops` | Seq Scan | 1650.08 | 15.2 ms |
| `payload ? 'tags'` | GIN `jsonb_ops` (1696 kB) | Bitmap Index → Heap Scan | 1232.51 | **10.8 ms** |
| avg total by payment_method | JSONB, 1003 buffers | Seq Scan + HashAggregate | 1954.22 | 17.7 ms |
| avg total by payment_method | normalized, 97 buffers | Seq Scan + HashAggregate | 269.07 | **7.7 ms** |

**Ціна запису:** `INSERT` 50 000 рядків без індексів займає 0.157 s, а з трьома JSONB-індексами — 0.792 s (**5× повільніше**).

Детальний аналіз — у розділі **Reflection** в кінці notebook.

## Відтворюваність

- Notebook виконується через **Restart & Run All** без ручних правок.
- Кожен SQL-блок починається з `DROP ... IF EXISTS`. View видаляється раніше за таблицю, від якої залежить, бо сервер PostgreSQL переживає рестарт kernel.
- Перед кожним виміром індекси явно видаляються, а після створення індексу виконується `ANALYZE`.
- Синтетичні дані та stub генеруються детерміновано, від фіксованих дат.
- Усі службові таблиці мають префікс `t7_hw_`.
