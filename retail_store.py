from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, coalesce, when, round, 
    sum, avg, desc, dayofweek, month
)

# 1. Загрузка и предварительная обработка данных

# Инициализация Spark-сессии
spark = SparkSession.builder \
    .appName("RetailStoreSales") \
    .getOrCreate()

# Путь к файлу
file_path = "c:/de/spark/retail_store_sales.csv"

# Загрузка данных
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Вывод первых 5 строк
print("Первые 5 строк DataFrame")
df.show(5, truncate=False)

# Вывод схемы
print("Схема DataFrame")
df.printSchema()

# 1.2. Преобразование названий столбцов в snake_case
for col_name in df.columns:
    new_col_name = col_name.strip().lower().replace(' ', '_')
    df = df.withColumnRenamed(col_name, new_col_name)

# Вывод обновленной схемы
print("Обновленная схема DataFrame (snake_case)")
df.printSchema()

# 1.3. Приведение типов данных
df = df.withColumn("price_per_unit", col("price_per_unit").cast("double")) \
       .withColumn("quantity", col("quantity").cast("integer")) \
       .withColumn("total_spent", col("total_spent").cast("double")) \
       .withColumn("transaction_date", to_date(col("transaction_date"), "yyyy-MM-dd")) \
       .withColumn("discount_applied", col("discount_applied").cast("boolean"))

# Вывод итоговой схемы
print("Схема DataFrame после приведения типов")
df.printSchema()

print("Проверка данных после преобразования")
df.show(5, truncate=False)

# 2. Очистка и валидация данных

# 2.1. Восстановление отсутствующих item
print("Восстановление item")

# Создаем справочник товаров
ref_df = df.select("category", "price_per_unit", "item") \
           .dropna(subset=["item", "category", "price_per_unit"]) \
           .dropDuplicates()

# Делаем left join основного датафрейма со справочником
# Условие join: совпадение категории и цены
df_joined = df.alias("main").join(
    ref_df.alias("ref"),
    (col("main.category") == col("ref.category")) & 
    (col("main.price_per_unit") == col("ref.price_per_unit")),
    how="left"
)

# Обновляем колонку item: если была null, берем из справочника
df = df_joined.select(
    *[col(f"main.{c}") for c in df.columns if c != "item"],
    coalesce(col("main.item"), col("ref.item")).alias("item")
)

# Выводим 20 строк, где item изначально отсутствовал, но теперь восстановлен
print("Восстановленные значения item:")
df.filter(col("item").isNotNull()).show(20, truncate=False)

# 2.2. Восстановление Total Spent
print("Восстановление total_spent")

# Пересчитываем total_spent там, где он null, а quantity и price_per_unit есть
df = df.withColumn(
    "total_spent",
    when(
        col("total_spent").isNull() & 
        col("quantity").isNotNull() & 
        col("price_per_unit").isNotNull(),
        col("quantity") * col("price_per_unit")
    ).otherwise(col("total_spent"))
)

# 2.3. Заполнение отсутствующих Quantity и Price Per Unit
print("Заполнение Quantity и Price Per Unit")

# Сохраним копию до изменений, чтобы потом показать именно восстановленные строки
df_before_23 = df 

# Восстанавливаем quantity (total_spent / price_per_unit), приводим к integer
df = df.withColumn(
    "quantity",
    when(
        col("quantity").isNull() & 
        col("total_spent").isNotNull() & 
        col("price_per_unit").isNotNull() & (col("price_per_unit") > 0),
        (col("total_spent") / col("price_per_unit")).cast("integer")
    ).otherwise(col("quantity"))
)

# Восстанавливаем price_per_unit (total_spent / quantity), округляем до 2 знаков
df = df.withColumn(
    "price_per_unit",
    when(
        col("price_per_unit").isNull() & 
        col("total_spent").isNotNull() & 
        col("quantity").isNotNull() & (col("quantity") > 0),
        round(col("total_spent") / col("quantity"), 2)
    ).otherwise(col("price_per_unit"))
)

# Выводим 20 строк, где значения были восстановлены (были null до шага 2.3)
print("Восстановленные значения quantity и price_per_unit:")
# Находим ID транзакций, где были пропуски до шага 2.3
ids_to_show = df_before_23.filter(col("quantity").isNull() | col("price_per_unit").isNull()).select("transaction_id")
df.join(ids_to_show, on="transaction_id", how="inner").show(20, truncate=False)

# 2.4. Удаление оставшихся строк с пропусками
print("Удаление оставшихся строк с пропусками")

# Удаляем строки, где остались null в ключевых колонках
df = df.dropna(subset=["category", "quantity", "total_spent", "price_per_unit"])

print(f"Итоговое количество строк после очистки: {df.count()}")
df.printSchema()

# 3. Разведочный анализ данных

# 3.1. Самые популярные категории товаров
print("Топ-5 категорий по общему количеству проданных единиц")

top_categories = df.groupBy("category") \
    .agg(sum("quantity").alias("total_quantity_sold")) \
    .orderBy(desc("total_quantity_sold")) \
    .limit(5)

top_categories.show(truncate=False)


# 3.2. Анализ среднего чека
print("Средний чек (Total Spent) по методу оплаты")

avg_payment_method = df.groupBy("payment_method") \
    .agg(round(avg("total_spent"), 2).alias("avg_total_spent")) \
    .orderBy(desc("avg_total_spent"))

avg_payment_method.show(truncate=False)


print("Средний чек (Total Spent) по локации")

avg_location = df.groupBy("location") \
    .agg(round(avg("total_spent"), 2).alias("avg_total_spent")) \
    .orderBy(desc("avg_total_spent"))

avg_location.show(truncate=False)

# 4. Генерация признаков 

# 4.1. Временные признаки
print("Добавление временных признаков")

# dayofweek: возвращает день недели от 1 (Воскресенье) до 7 (Суббота)
# month: возвращает номер месяца от 1 до 12
df = df.withColumn("day_of_week", dayofweek(col("transaction_date"))) \
       .withColumn("transaction_month", month(col("transaction_date")))

# Продажи по дням недели
print("Средняя сумма продаж по дням недели")

# Группируем по дню недели, считаем среднее, округляем и сортируем
sales_by_day = df.groupBy("day_of_week") \
    .agg(round(avg("total_spent"), 2).alias("avg_total_spent")) \
    .orderBy("day_of_week")

sales_by_day.show(truncate=False)

# Продажи по месяцам
print("Средняя сумма продаж по месяцам")

# Группируем по месяцу, считаем среднее, округляем и сортируем
sales_by_month = df.groupBy("transaction_month") \
    .agg(round(avg("total_spent"), 2).alias("avg_total_spent")) \
    .orderBy("transaction_month")

sales_by_month.show(truncate=False)

# Признаки клиента (CLV)
print("Топ-10 клиентов по Customer Lifetime Value (CLV)")

# Группируем по клиенту, суммируем все его траты, сортируем по убыванию и берем топ-10
clv_df = df.groupBy("customer_id") \
    .agg(round(sum("total_spent"), 2).alias("customer_lifetime_value")) \
    .orderBy(desc("customer_lifetime_value")) \
    .limit(10)

clv_df.show(truncate=False)