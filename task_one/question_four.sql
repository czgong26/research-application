-- QUESTION 4: Retrieves Year and Score from the score table
-- Orders the happiness scores of each year in descending order and each year in ascending order
SELECT "Year", Score
FROM score
WHERE Score IS NOT NULL
GROUP BY Score
ORDER BY "Year" ASC, Score DESC;
