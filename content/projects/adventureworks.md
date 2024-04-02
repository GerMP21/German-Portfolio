---
title: "AdventureWorks Sales Analysis and Dashboard"
type: page
---

[Github Project](https://github.com/GerMP21/AdventureWorks-Sales-Analysis)

![Sales Overview](/images/adventureworks/salesoverview.png "Sales Overview")

## Business Request & User Stories
The business request for this project was an executive sales report for sales managers. Based on the request that was made from the business we following user stories were defined to fulfill delivery and ensure that acceptance criteria’s were maintained throughout the project.

|     As a (role)      |              I want (request / demand)              |                          So that I can (user value)                          |                             Acceptance Criteria                             |
| -------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Sales Manager        | A dashboard overview of internet sales              | Follow sales over time against budget                                        | A Power Bi dashboard with graphs and KPIs comparing sales against budget.   |
| Sales Manager        | A dashboard overview of internet sales              | Follow better which customers buy the most and which products sells the best | A Power BI dashboard with the top selling products and top buying customers |
| Sales Representative | A detailed overview of internet sales per customers | Can follow up my customers that buys the most                                | A Power BI dashboard which allows me to filter data for each customer       |
| Sales Representative | A detailed overview of internet sales per products  | Can follow up my products that sells the most                                | A Power BI dashboard which allows me to filter data for each product        |

## Data Cleansing & Transformation (SQL)
To create the necessary data model for doing analysis and fulfilling the business needs defined in the user stories the following tables were extracted using SQL.

One data source (sales budgets) was provided in Excel format and were connected in the data model in a later step of the process.

Below are the SQL statements for cleansing and transforming necessary data.

**DimDate:**
``` sql
-- Cleansed DIM_Date Table --
SELECT 
  [DateKey], 
  [FullDateAlternateKey] AS Date, 
  [EnglishDayNameOfWeek] AS Day, 
  [EnglishMonthName] AS Month, 
  Left([EnglishMonthName], 3) AS MonthShort,
  [MonthNumberOfYear] AS MonthNo, 
  [CalendarQuarter] AS Quarter, 
  [CalendarYear] AS Year
FROM 
 [AdventureWorksDW2022].[dbo].[DimDate]
WHERE 
  CalendarYear >= 2022
  AND CalendarYear < 2024
```

**DimCustomer:**
``` sql
-- Cleansed DIM_Customers Table --
SELECT 
  c.customerkey AS CustomerKey, 
  c.firstname AS [First Name], 
  c.lastname AS [Last Name], 
  c.firstname + ' ' + lastname AS [Full Name],
  CASE c.gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' END AS Gender,
  c.datefirstpurchase AS DateFirstPurchase, 
  g.city AS [Customer City] -- Joined in Customer City from Geography Table
FROM 
  [AdventureWorksDW2022].[dbo].[DimCustomer] as c
  LEFT JOIN [AdventureWorksDW2022].[dbo].[DimGeography] AS g ON g.geographykey = c.geographykey 
ORDER BY 
  CustomerKey ASC -- Ordered List by CustomerKey
```

**DimProduct:**
``` sql
-- Cleansed DIM_Products Table --
SELECT 
  p.[ProductKey], 
  p.[ProductAlternateKey] AS ProductItemCode, 
  p.[EnglishProductName] AS [Product Name], 
  psc.EnglishProductSubcategoryName AS [Sub Category], -- Joined in from Sub Category Table
  pc.EnglishProductCategoryName AS [Product Category], -- Joined in from Category Table
  p.[Color] AS [Product Color], 
  p.[Size] AS [Product Size], 
  p.[ProductLine] AS [Product Line], 
  p.[ModelName] AS [Product Model Name], 
  p.[EnglishDescription] AS [Product Description], 
  ISNULL (p.Status, 'Outdated') AS [Product Status] 
FROM 
  [AdventureWorksDW2022].[dbo].[DimProduct] as p
  LEFT JOIN [AdventureWorksDW2022].[dbo].[DimProductSubcategory] AS psc ON psc.ProductSubcategoryKey = p.ProductSubcategoryKey 
  LEFT JOIN [AdventureWorksDW2022].[dbo].[DimProductCategory] AS pc ON psc.ProductCategoryKey = pc.ProductCategoryKey 
order by 
  p.ProductKey asc
```

**FactInternetSales:**
``` sql
-- Cleansed FACT_InternetSales Table --
SELECT 
  [ProductKey], 
  [OrderDateKey], 
  [DueDateKey], 
  [ShipDateKey], 
  [CustomerKey], 
  [SalesOrderNumber], 
  [SalesAmount]
FROM 
  [AdventureWorksDW2022].[dbo].[FactInternetSales]
WHERE 
  LEFT (OrderDateKey, 4) >= YEAR(GETDATE()) - 2 -- Ensures we always only bring two years of date from extraction
  AND LEFT (OrderDateKey, 4) < '2024' -- Filter out sales before 2024
ORDER BY
  OrderDateKey ASC
```

## Data Model
Below is a screenshot of the data model after cleansed and prepared tables were read into Power BI.

![Data Model](/images/adventureworks/datamodel.png "Data Model")

## Sales Management Dashboard
The finished sales management dashboard with one page with works as a dashboard and overview, with two other pages focused on combining tables for necessary details and visualizations to show sales over time, per customers and per products.

![Sales Overview](/images/adventureworks/salesoverview.png "Sales Overview")

![Product Details](/images/adventureworks/productdetails.png "Product Details")

![Customer Details](/images/adventureworks/customerdetails.png "Customer Details")