---
title: "AdventureWorks Sales Analysis and Dashboard"
type: page
---

![Sales Overview](/images/adventureworks/salesoverview.png "Sales Overview")

![Product Details](/images/adventureworks/productdetails.png "Product Details")

![Customer Details](/images/adventureworks/customerdetails.png "Customer Details")

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



