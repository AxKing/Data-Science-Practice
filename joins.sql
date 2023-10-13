-- SQL notes

-- INNER JOIN
SELECT p1.country, p1.continent, prime_minister, president
FROM prime_ministers AS p1
INNER JOIN presidents AS p2
ON p1.country = p2.country
;

-- IF the joining column is the same in both tables, you can use USING
SELECT left_table.id AS L_id,
	left_table.val AS L_val,
	right_table.cal AS R_val
FROM left_table
INNER JOIN right_table
USING (id); --Parenthesis are required here for the join.

-- CASE, WHEN, and THEN
SELECT name, continent, indep_year,
	CASE WHEN indep_year < 1900 THEN 'before 1900'
		WHEN indep_year <= 1930 THEN 'between 1900 and 1930'
		ELSE 'after 1930' END
		AS indep_year_group
FROM states
ORDER BY indep_year_group
;

-- OUTER JOINS

--CROSS JOIN

-- UNION: Doesn't include duplicates
-- UNION ALL: includes Duplicates
-- INTERSECT:
-- EXCEPT:

SELECT prime_minster AS leader, country
FROM prime_ministers

UNION

SELECT monarch, country
FROM monarchs

ORDER BY country


-- SEMI-JOIN (using subqueries)
SELECT president, country, continent
FROM presidents
WHERE country IN
	(SELECT name
		FROM states
		WHERE indep_year < 1800)
;

--ANTI-JOIN
SELECT president, country, continent
FROM presidents
WHERE continent LIKE '%America'
	AND country NOT IN
	(SELECT name
		FROM states
		WHERE indep_year < 1800)



SELECT DISTINCT monarchs.continent, subquery.max_perc
FROM monarchs,
	(SELECT continent, MAX(women_parli_perc) AS max_perc
		FROM states
		GROUP BY continent) as subquery
WHERE monarchs.continent = subquery.continent
ORDER BY continent
;

-- Select fields
SELECT countries.local_name, subquery.lang_num
  -- From countries
  FROM countries, 
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 FROM languages
  	 GROUP BY code) AS subquery
  -- Where codes match
  WHERE subquery.code = countries.code
-- Order by descending number of languages
ORDER BY lang_num DESC;


LIKE
--
NOT LIKE
IN







