-- QUESTION 1: Retrieves Country, Year, and the sum of a country’s GDP per year from gdp table
SELECT
	Country, "Year", sum(GDP)
AS
	GDP_sum
FROM
	gdp
GROUP BY
	Country, "Year";
