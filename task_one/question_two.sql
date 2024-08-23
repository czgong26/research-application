-- QUESTION 2: Retrieves Country, Year, and a country’s yearly average happiness score from score table
SELECT Country, "Year", avg(Score) AS Avg_score FROM score
GROUP BY Country, "Year";