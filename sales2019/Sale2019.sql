USE SALES
SELECT *
FROM sales2019

-- thay doi ten cot vi ten cot chua khoang trang
EXEC sp_rename 'sales2019."Order ID"', 'OrderID', 'COLUMN';
EXEC sp_rename 'sales2019."Quantity Ordered"', 'QuantityOrdered', 'COLUMN';
EXEC sp_rename 'sales2019."Order Date"', 'OrderDate', 'COLUMN';
EXEC sp_rename 'sales2019."Price Each"', 'PriceEach', 'COLUMN';
EXEC sp_rename 'sales2019."Purchase Address"', 'PurchaseAddress', 'COLUMN';
-- lay gia tri thang neu khong dua vao cot month co san 
SELECT DATEPART(mm,OrderDate)
FROM sales2019

--tim thang co doanh thu tot nhat What was the best month for sales? How much was earned that month?

-- thay doi kieu du lieu cua cot tu varchar ==> int
ALTER TABLE sales2019
ALTER COLUMN QuantityOrdered INT;

ALTER TABLE sales2019
ALTER COLUMN PriceEach FLOAT;

-- tinh tong doanh thu theo thang nhung khong them cot sales vao bang
SELECT DATEPART(mm,OrderDate) AS Month,
	   SUM(QuantityOrdered*PriceEach) AS Sales_all_month
FROM sales2019
GROUP BY DATEPART(mm,OrderDate)
ORDER BY 2 DESC;

-- them cot Sales vao bang 
ALTER TABLE sales2019
ADD Sales FLOAT;

--xoa cot ALTER TABLE sales2019 DROP COLUMN Sales;

UPDATE sales2019
SET Sales = QuantityOrdered*PriceEach;

-- tinh tong doanh thu cua tung thang nhung them cot sales vao bang 
SELECT DATEPART(mm,OrderDate) AS Month,
	   SUM(Sales) AS Sales_all_month
FROM sales2019
GROUP BY DATEPART(mm,OrderDate);

--Tìm thành phố có doanh thu cao nhất
-- Lấy tên thành phố từ cột PurchaseAddress

WITH city AS (SELECT  OrderID, PurchaseAddress, value AS CITY
FROM sales2019
CROSS APPLY STRING_SPLIT(PurchaseAddress,',',1)
WHERE ordinal = 2)

SELECT c.CITY, SUM(s.Sales) Sales
FROM sales2019 s
JOIN city c
ON c.OrderID = s.OrderID
GROUP BY c.CITY
ORDER BY 2 DESC;
-- ==> Thành phố San Francisco là thành phố có doanh thu cao nhất

-- Doanh nghiệp cần tăng chiếu quảng cáo vào khung thời gian nào
-- để tăng khả năng mua hàng của khách hàng

SELECT DATEPART(hh,OrderDate), SUM(QuantityOrdered)
FROM sales2019
GROUP BY DATEPART(hh,OrderDate)
ORDER BY 2 DESC;
-- ==> khung thời gian 19 và 12 giờ là có nhiều đơn hàng nhất

-- Những đơn hàng có số lượng sản phẩm >=2 , giá riêng của từng sản phẩm và giá của cả đơn hàng
SELECT OrderID, Product, Sales, 
	  SUM(Sales) OVER (PARTITION BY OrderID) AS Sales_by_OrderID
FROM sales2019
WHERE OrderID IN (SELECT OrderID
				  FROM sales2019
				  GROUP BY OrderID
				  HAVING COUNT(OrderID)>1);

-- tạo bảng đơn giá cho từng sản phẩm 
SELECT DISTINCT Product, PriceEach 
INTO PriceProduct
FROM sales2019;

-- Sản phẩm bán chạy nhất  : AAA Batteries(4-pack) cũng có thể vì giá sản phẩm rẻ
SELECT p.Product, p.PriceEach, SUM(s.QuantityOrdered) QuantityOrdered
FROM sales2019 s
JOIN PriceProduct p
ON p.Product = s.Product
GROUP BY p.Product, p.PriceEach
ORDER BY 3 DESC

-- Ngày nào có doanh thu cao nhất : 4/12/2019 doanh thu= 166727
SELECT  TOP 1 DATETRUNC(dd, OrderDate) AS Day, SUM(Sales) Sales
FROM sales2019
GROUP BY DATETRUNC(dd, OrderDate)
ORDER BY 2 DESC;

SELECT *
FROM (SELECT  DATETRUNC(dd, OrderDate) AS Day, SUM(Sales) Sales
	  FROM sales2019
	  GROUP BY DATETRUNC(dd, OrderDate)) sub
WHERE DATETRUNC(dd, sub.Day) = '2019-11-29'
--BlackFriday năm 2019 là ngày 29/11 nhưng chỉ có doanh thu = 92666
 




