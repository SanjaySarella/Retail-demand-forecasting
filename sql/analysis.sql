-- Portable analyst workflow for the retail case study.
SELECT Store, Dept, Date, Weekly_Sales
FROM read_csv_auto('data/raw/train.csv')
QUALIFY COUNT(*) OVER (PARTITION BY Store, Dept, Date) = 1;

-- Data-quality summary.
SELECT COUNT(*) AS source_rows,
       COUNT(*) - COUNT(DISTINCT (Store, Dept, Date)) AS duplicate_keys,
       SUM(CASE WHEN Weekly_Sales < 0 THEN 1 ELSE 0 END) AS negative_sales_rows
FROM read_csv_auto('data/raw/train.csv');

-- Weekly series used by the application; parameterize Store and Dept.
SELECT Date, Weekly_Sales
FROM read_csv_auto('data/raw/train.csv')
WHERE Store = 1 AND Dept = 1
ORDER BY Date;
