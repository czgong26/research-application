-- QUESTION 3: Joins the gdp, score, and region tables together
-- Countries that do not have a specified region are labeled as ‘null region’
SELECT gdp.Country, COALESCE(region.Region, 'null region') as Region, gdp."Year", gdp.gdp, score.Score
FROM gdp
JOIN score
	ON gdp.Country = score.Country
	AND gdp."Year" = score."Year"
	AND gdp.State_id = score.State_id
LEFT JOIN region
	ON gdp.Country = region.Country
WHERE gdp.Country IS NOT NULL;
