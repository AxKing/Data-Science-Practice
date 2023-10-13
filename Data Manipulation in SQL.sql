-- Data manipulation in SQL

/* 
Tables we have are
country, league, team and match
*/

-----------------------------------------------------------------------------------------------------------
----------------------------------------------------CHATPER 1----------------------------------------------
-----------------------------------------------------------------------------------------------------------

-- Chapter 1
-- Section 1
-- We'll take the CASE

-- Intro stuff to know
SELECT user_id, SUM(sales)
FROM sales_data
WHERE user_id BETWEEN 300 AND 400
GROUP BY user_id
;

SELECT c.counttry, c.team, SUM(m.goals)
FROM countries AS c
LEFT JOIN matches AS m
ON c.team_id = m.home_team_id
WHERE m.year >1990
GROUP BY c.country, c.team
;

-- European Soccer Database
SELECT 
	l.name AS league,
	COUNT(m.country_id) AS total_matches
FROM league AS L
LEFT JOIN match AS m
ON l.country_id = m.country_id
GROUP BY l.name
;

SELECT
	date,
	id,
	home_goal,
	away_goal
FROM match
WHERE season = '2013/2014'
	AND home_team_goal > away_team_goal;
;

-- CASE statement
WHEN, THEN, ELSE finishing with END

CASE 	WHEN x =1 	THEN 'a'
		WHEN x = 2 	THEN 'b'
		ELSE 'c' END as new_column

SELECT
	id,
	home_goal,
	away_goal,
	CASE 	WHEN home_goal > away_goal THEN 'Home Team Win'
			WHEN home_goal < away_goal THEN 'Away Team Win'
			ELSE 'Tie' END as outcome
FROM match
WHERE season = '2013/2014'



-- Chapter 1
-- Section 1
-- Exercises

SELECT
	-- Select the team long name and team API id
	team_api_id,
	team_long_name
FROM teams_germany
-- Only include FC Schalke 04 and FC Bayern Munich
WHERE team_long_name IN ('FC Bayern Munich', 'FC Schalke 04')
;

-- Identify the home team as Bayern Munich, Schalke 04, or neither
SELECT 
	CASE WHEN hometeam_id = 10189 THEN 'FC Schalke 04'
        WHEN hometeam_id = 9823 THEN 'FC Bayern Munich'
         ELSE 'Other' END AS home_team,
	COUNT(id) AS total_matches
FROM matches_germany
-- Group by the CASE statement alias
GROUP BY home_team
;


SELECT 
	-- Select the date of the match
	date,
	-- Identify home wins, losses, or ties
	CASE WHEN home_goal > away_goal THEN 'Home win!'
        WHEN home_goal < away_goal THEN 'Home loss :(' 
        ELSE 'Tie' END AS outcome
FROM matches_spain
;
SELECT 
	m.date,
	--Select the team long name column and call it 'opponent'
	t.team_long_name AS opponent, 
	-- Complete the CASE statement with an alias
	CASE WHEN m.home_goal > m.away_goal THEN 'Home win!'
        WHEN m.home_goal < m.away_goal THEN 'Home loss :('
        ELSE 'Tie' END AS outcome
FROM matches_spain AS m
-- Left join teams_spain onto matches_spain
LEFT JOIN teams_spain AS t
ON m.awayteam_id = t.team_api_id
;
SELECT 
	m.date,
	t.team_long_name AS opponent,
    -- Complete the CASE statement with an alias
	CASE WHEN m.home_goal > m.away_goal THEN 'Barcelona win!'
        WHEN m.home_goal < m.away_goal THEN 'Barcelona loss :(' 
        ELSE 'Tie' END AS outcome 
FROM matches_spain AS m
LEFT JOIN teams_spain AS t 
ON m.awayteam_id = t.team_api_id
-- Filter for Barcelona as the home team
WHERE m.hometeam_id = 8634
; 
-- Select matches where Barcelona was the away team
SELECT  
	m.date,
	t.team_long_name AS opponent,
	CASE WHEN m.home_goal < m.away_goal THEN 'Barcelona win!'
        WHEN m.home_goal > m.away_goal  THEN 'Barcelona loss :(' 
        ELSE 'Tie' END AS outcome
FROM matches_spain AS m
-- Join teams_spain to matches_spain
LEFT JOIN teams_spain AS t 
ON m.hometeam_id = t.team_api_id
WHERE m.awayteam_id = 8634
;


-- Chapter 1
-- Section 2
-- In CASE things get more complex

SELECT date, hometeam_id, awayteam_id,
	CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
		THEN 'Chelsea home win!'
	WHEN awayteam_id = 8455 AND home_goal < away_goal
		THEN 'Chelsea away win!'
	ELSE 'Loss or Tie :(' END AS outcome
FROM match
WHERE hometeam_id = 8455 OR awayteam_id = 8455
;

SELECT date,
	CASE WHEN date > '2015-01-01' THEN 'More Recently'
		WHEN date < '2012-01-01' THEN 'Older'
		END AS date_category
FROM match
;
SELECT date,
	CASE WHEN date > '2015-01-01' THEN 'More Recently'
		WHEN date < '2012-01-01' THEN 'Older'
		ELSE NULL END AS date_category
FROM match
;

SELECT date, season,
	CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
		THEN 'Chelsea home win!'
	CASE WHEN awayteam_id = 8455 AND home_goal < away_goal
		THEN 'Chelsea away win!'
	END as outcome
FROM matches_spain
WHERE CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
		THEN 'Chelsea home win!'
	CASE WHEN awayteam_id = 8455 AND home_goal < away_goal
		THEN 'Chelsea away win!'
	END IS NOT NULL
;
-- Chapter 1
-- Section 2
-- Exercises

SELECT 
	date,
	-- Identify the home team as Barcelona or Real Madrid
	CASE WHEN hometeam_id = 8634 THEN 'FC Barcelona' 
        WHEN hometeam_id = 8633 THEN 'Real Madrid CF' END AS home,
    -- Identify the away team as Barcelona or Real Madrid
	CASE WHEN awayteam_id = 8634 THEN 'FC Barcelona' 
        WHEN awayteam_id = 8633 THEN 'Real Madrid CF' END AS away
FROM matches_spain
WHERE (awayteam_id = 8634 OR hometeam_id = 8634)
      AND (awayteam_id = 8633 OR hometeam_id = 8633)
;
SELECT 
	date,
	CASE WHEN hometeam_id = 8634 THEN 'FC Barcelona' 
         ELSE 'Real Madrid CF' END as home,
	CASE WHEN awayteam_id = 8634 THEN 'FC Barcelona' 
         ELSE 'Real Madrid CF' END as away,
	-- Identify all possible match outcomes
	CASE WHEN home_goal > away_goal AND hometeam_id = 8634 THEN 'Barcelona win!'
        WHEN home_goal > away_goal AND hometeam_id = 8633 THEN 'Real Madrid win!'
        WHEN home_goal < away_goal AND awayteam_id = 8634 THEN 'Barcelona win!'
        WHEN home_goal < away_goal AND awayteam_id = 8633 THEN 'Real Madrid win!'
        ELSE 'Tie!' END AS outcome
FROM matches_spain
WHERE (awayteam_id = 8634 OR hometeam_id = 8634)
      AND (awayteam_id = 8633 OR hometeam_id = 8633)
;


-- Select team_long_name and team_api_id from team
SELECT
	team_long_name,
	team_api_id
FROM team
-- Filter for team long name
WHERE team_long_name = 'Bologna'
;
-- Select the season and date columns
SELECT 
	season,
	date,
    -- Identify when Bologna won a match
	CASE WHEN hometeam_id = 9857
        AND home_goal > away_goal 
        THEN 'Bologna Win'
		WHEN awayteam_id = 9857 
        AND away_goal > home_goal 
        THEN 'Bologna Win' 
		END AS outcome
FROM matches_italy
;
-- Select the season, date, home_goal, and away_goal columns
SELECT 
	season,
    date,
	home_goal,
	away_goal
FROM matches_italy
WHERE 
-- Exclude games not won by Bologna
	CASE WHEN hometeam_id = 9857 AND home_goal > away_goal THEN 'Bologna Win'
	 	WHEN awayteam_id = 9857 AND away_goal > home_goal THEN 'Bologna Win' 
	END IS NOT NULL;


-- Chapter 1
-- Section 3
-- CASE WHEN with aggregate functions

-- categorizing data
-- filtering data
-- aggregating data

-- Suppose we wanted to count home goals and away goals
SELECT
	season,
	COUNT(CASE WHEN hometeam_id = 8650
			AND home_goal > away_goal
			THEN id END) AS home_wins
	COUNT(CASE WHEN awayteam_id = 8650
			AND away_goal > home_goal
			THEN id END) as away_wins
FROM match
GROUP BY season
;

-- Suppose we want sum the number of goals
SELECT
	season,
	SUM(CASE WHEN hometeam_id = 8650
			THEN home_goal END) as home_goals,
	SUM(CASE WHEN awayteam_id = 8650
			THEN away_goal END) as away_goals
FROM match
GROUP BY season
;

-- Using the Average Function
SELECT
	season,
	AVG(CASE WHEN hometeam_id = 8650
			THEN home_goal END) as home_goals,
	AVG(CASE WHEN awayteam_id = 8650
			THEN away_goal END) as away_goals
FROM match
GROUP BY season
;

ROUND(3.141592653589, 2)
--Returns 3.14


SELECT
	season,
	ROUND(AVG(CASE WHEN hometeam_id = 8650
	 			THEN home_goal END), 2) as home_goals,
	ROUND(AVG(CASE WHEN awayteam_id = 8650
	 			THEN away_goal END), 2) as away_goals
FROM match
GROUP BY season
;


--Percentages with CASE and AVG
-- What percentage of liverpool games did they win this season?
SELECT
	season,
	AVG(CASE WHEN hometeam_id = 8455 AND home_goal > away_goal THEN 1
		WHEN hometeam_id = 8455 AND homegoal < away_goal THEN 0
		END) as pct_homewins,
	AVG(CASE WHEN awayteam_id = 8455 AND away_goal > home_goal THEN 1
		WHEN awayteam_id = 8455 AND away_goal < home_goal THEN 0
		END) as pct_awaywins
FROM match
GROUP BY season


SELECT
	season,
	ROUND(AVG(CASE WHEN hometeam_id = 8455 AND home_goal > away_goal THEN 1
	 		WHEN hometeam_id = 8455 AND homegoal < away_goal THEN 0
	 		END), 2) as pct_homewins,
	ROUND(AVG(CASE WHEN awayteam_id = 8455 AND away_goal > home_goal THEN 1
	 		WHEN awayteam_id = 8455 AND away_goal < home_goal THEN 0
	 		END), 2) as pct_awaywins
FROM match
GROUP BY season


-- Chapter 1
-- Section 3
-- Exercises
SELECT 
	c.name AS country,
    -- Count games from the 2012/2013 season
	COUNT(CASE WHEN m.season = '2012/2013' 
        	THEN m.id ELSE NULL END) AS matches_2012_2013
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY c.name
;
SELECT 
	c.name AS country,
    -- Count matches in each of the 3 seasons
	COUNT(CASE WHEN m.season = '2012/2013' THEN m.id END) AS matches_2012_2013,
	COUNT(CASE WHEN m.season = '2013/2014' THEN m.id END) AS matches_2013_2014,
	COUNT(CASE WHEN m.season = '2014/2015' THEN m.id END) AS matches_2014_2015
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY c.name
;

SELECT 
	c.name AS country,
    -- Sum the total records in each season where the home team won
	SUM(CASE WHEN m.season = '2012/2013' AND m.home_goal > m.away_goal 
        THEN 1 ELSE 0 END) AS matches_2012_2013,
 	SUM(CASE WHEN m.season = '2013/2014' AND m.home_goal > m.away_goal 
        THEN 1 ELSE 0 END) AS matches_2013_2014,
	SUM(CASE WHEN m.season = '2014/2015' AND m.home_goal > m.away_goal 
        THEN 1 ELSE 0 END) AS matches_2014_2015
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY c.name
;

SELECT 
    c.name AS country,
    -- Count the home wins, away wins, and ties in each country
	COUNT(CASE WHEN m.home_goal > m.away_goal THEN m.id 
        END) AS home_wins,
	COUNT(CASE WHEN m.home_goal < m.away_goal THEN m.id 
        END) AS away_wins,
	COUNT(CASE WHEN m.home_goal = m.away_goal THEN m.id 
        END) AS ties
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country
;
SELECT 
	c.name AS country,
    -- Calculate the percentage of tied games in each season
	AVG(CASE WHEN m.season='2013/2014' AND m.home_goal = m.away_goal THEN 1
			WHEN m.season='2013/2014' AND m.home_goal != m.away_goal THEN 0
			END) AS ties_2013_2014,
	AVG(CASE WHEN m.season='2014/2015' AND m.home_goal = m.away_goal THEN 1
			WHEN m.season='2014/2015' AND m.home_goal != m.away_goal THEN 0
			END) AS ties_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country
;
SELECT 
	c.name AS country,
    -- Round the percentage of tied games to 2 decimal points
	ROUND(AVG(CASE WHEN m.season='2013/2014' AND m.home_goal = m.away_goal THEN 1
			 WHEN m.season='2013/2014' AND m.home_goal != m.away_goal THEN 0
			 END),2) AS pct_ties_2013_2014,
	ROUND(AVG(CASE WHEN m.season='2014/2015' AND m.home_goal = m.away_goal THEN 1
			 WHEN m.season='2014/2015' AND m.home_goal != m.away_goal THEN 0
			 END),2) AS pct_ties_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country
;

-----------------------------------------------------------------------------------------------------------
----------------------------------------------------CHATPER 2----------------------------------------------
-----------------------------------------------------------------------------------------------------------

-- Chapter 2
-- Section 1
-- WHERE are the subqueries?

SELECT column
FROM (SELECT column)
	FROM table) as subquery
;
-- Can be used in SELECT, FROM, WHERE, and GROUP BY
-- Can return numbers, a list or a table
SELECT home_goal
FROM match
WHERE home_goal > (SELECT AVG(home_goal) 
					FROM match)
;

-- WHERE clause is helpful for filtering results.
-- 2012/2013 where number of goals was higher than average.
SELECT date,hometeam_id, awayteam_id, home_goal, away_goal
FROM match
WHERE season = '2012/2013'
AND home_goal > (SELECT AVG(home_goal) FROM match)
;

-- Filtering a list with IN
SELECT team_long_name, team_short_name as abbr
FROM team
WHERE
	team_api_id IN 
		(SELECT hometeam_id
		FROM match
		WHERE country_id = 15722)
;

-- Chapter 2
-- Section 1
-- Exercises

-- Select the average of home + away goals, multiplied by 3
SELECT 
	3 * AVG(home_goal + away_goal)
FROM matches_2013_2014
;

SELECT 
	-- Select the date, home goals, and away goals scored
    date,
	home_goal,
	away_goal
FROM  matches_2013_2014
-- Filter for matches where total goals exceeds 3x the average
WHERE (home_goal + away_goal) > 
       (SELECT 3 * AVG(home_goal + away_goal)
        FROM matches_2013_2014)
;
SELECT 
	-- Select the team long and short names
	team_long_name,
	team_short_name
FROM team 
-- Exclude all values from the subquery
WHERE team_api_id NOT IN
     (SELECT DISTINCT hometeam_ID  FROM match)
;
SELECT
	-- Select the team long and short names
	team_long_name,
	team_short_name
FROM team
-- Filter for teams with 8 or more home goals
WHERE team_api_id IN
	  (SELECT hometeam_ID 
       FROM match
       WHERE home_goal >= 8)
;

-- Chapter 2
-- Section 2
-- Subqueries in FROM

--Home goal average
SELECT
	t.team_long_name AS team,
	AVG(m.home_goal) AS home_avg
FROM match AS m
LEFT JOIN team as t
ON m.hometeam_id = t.team_api_id
WHERE season = '2011/2012'
GROUP BY team
;

-- Making it a subquery
SELECT team, home_avg
FROM (SELECT
	t.team_long_name AS team,
	AVG(m.home_goal) AS home_avg
FROM match AS m
LEFT JOIN team as t
ON m.hometeam_id = t.team_api_id
WHERE season = '2011/2012'
GROUP BY team
;) as subquery
ORDER BY home_avg DESC
LIMIT 3
;

-- You can create multiple subqueries in one FROM statement
-- alias them
-- join them!
-- You can join a subquery to a table in FROM
	--include a joining columns in both tables!


-- Chapter 2
-- Section 2
-- Exercises
SELECT 
	-- Select the country ID and match ID
	country_id, 
    id 
FROM match
-- Filter for matches with 10 or more goals in total
WHERE (home_goal + away_goal) >= 10
;

SELECT
	-- Select country name and the count match IDs
    c.name AS country_name,
    COUNT(sub.id) AS matches
FROM country AS c
-- Inner join the subquery onto country
-- Select the country id and match id columns
INNER JOIN (SELECT country_id, id
           FROM match
           -- Filter the subquery by matches with 10+ goals
           WHERE (home_goal + away_goal) >= 10) AS sub
ON c.id = sub.country_id
GROUP BY country_name
;

SELECT
	-- Select country, date, home, and away goals from the subquery
    country,
    date,
    home_goal,
    away_goal
FROM 
	-- Select country name, date, home_goal, away_goal, and total goals in the subquery
	(SELECT c.name AS country, 
     	    m.date, 
     		m.home_goal, 
     		m.away_goal,
           (m.home_goal + m.away_goal) AS total_goals
    FROM match AS m
    LEFT JOIN country AS c
    ON m.country_id = c.id) AS subquery
-- Filter by total goals scored in the main query
WHERE total_goals >= 10
;

-- Chapter 2
-- Section 3
-- Subqueries in SELECT

-- SELECTing what?
-- returns a single value
-- complex mathematical calculations

-- Calculate the total matches across all seasons.
--1.
SELECT COUNT(id) FROM match;

--2. ADD the sub query to the SELECT statement of the query
SELECT season, count(id) AS matches,
(SELECT COUNT(id) FROM match) as total_matches
FROM match
GROUP BY season;

-- Good for Math stuff too
SELECT AVG(home_goal + away_goal)
FROM match
WHERE season = '2011/2012'

-- To find the difference from the average for a given match
-- There are 2 ways
--.1
SELECT
	date,
	(home_goal + away_goal) as goals
	(home_goal + away_goal) - 2.72 as diff
	from match
	WHERE season='2011/2012'
;
--2. Or you can use a subquery
SELECT
	date,
	(home_goal + away_goal) AS goals,
	(home_goal + away_goal) - (
		SELECT AVG(home_goal+away_goal)
		FROM match
		WHERE season = '2011/2012') AS diff
FROM match
WHERE season = '2011/2012'
;

-- Things to keep in mind
-- needs to return a single value, not multiple rows
-- the information would need to be applied to each row in the table

-- Chapter 2
-- Section 3
-- Exercises

SELECT 
	l.name AS league,
    -- Select and round the league's total goals
    ROUND(AVG(m.home_goal + m.away_goal), 2) AS avg_goals,
    -- Select & round the average total goals for the season
    (SELECT ROUND(AVG(home_goal + away_goal), 2) 
     FROM match
     WHERE season = '2013/2014') AS overall_avg
FROM league AS l
LEFT JOIN match AS m
ON l.country_id = m.country_id
-- Filter for the 2013/2014 season
WHERE season = '2013/2014'
GROUP BY league
;


SELECT
	-- Select the league name and average goals scored
	l.name AS league,
	ROUND(AVG(m.home_goal + m.away_goal),2) AS avg_goals,
    -- Subtract the overall average from the league average
	ROUND(AVG(m.home_goal + m.away_goal) -
		(SELECT AVG(home_goal + away_goal)
		 FROM match 
         WHERE season = '2013/2014'),2) AS diff
FROM league AS l
LEFT JOIN match AS m
ON l.country_id = m.country_id
-- Only include 2013/2014 results
WHERE season = '2013/2014'
GROUP BY l.name
;



-- Chapter 2
-- Section 4
-- Subqueries everywhere! And best practices!

-- Multiple sub queries
-- Line up SELECT FROM WHERE and GROUP BY and what they're selecting
-- Annotate your queries with comments.
/* Use the long version
with multiple lines */

-- Indent your sub queries
--make sure you clearly indent all information that's within a single column

-- each subquery you add requires more computing power

-- watch properly placing your filters


-- Chapter 2
-- Section 4
-- Exercises
SELECT 
	-- Select the stage and average goals for each stage
	m.stage,
    ROUND(AVG(m.home_goal + m.away_goal),2) AS avg_goals,
    -- Select the average overall goals for the 2012/2013 season
    ROUND((SELECT AVG(home_goal + away_goal) 
           FROM match 
           WHERE season = '2012/2013'),2) AS overall
FROM match AS m
-- Filter for the 2012/2013 season
WHERE season = '2012/2013'
-- Group by stage
GROUP BY stage
;

SELECT 
	-- Select the stage and average goals from the subquery
	s.stage,
	ROUND(avg_goals,2) AS avg_goals
FROM 
	-- Select the stage and average goals in 2012/2013
	(SELECT
		 stage,
         AVG(home_goal + away_goal) AS avg_goals
	 FROM match
	 WHERE season = '2012/2013'
	 GROUP BY stage) AS s
WHERE 
	-- Filter the main query using the subquery
	s.avg_goals > (SELECT AVG(home_goal + away_goal) 
                    FROM match WHERE season = '2012/2013')
;

SELECT 
	-- Select the stage and average goals from s
	stage,
    ROUND(s.avg_goals,2) AS avg_goal,
    -- Select the overall average for 2012/2013
    (SELECT AVG(home_goal + away_goal) FROM match WHERE season = '2012/2013') AS overall_avg
FROM 
	-- Select the stage and average goals in 2012/2013 from match
	(SELECT
		 stage,
         AVG(home_goal + away_goal) AS avg_goals
	 FROM match
	 WHERE season = '2012/2013'
	 GROUP BY stage) AS s
WHERE 
	-- Filter the main query using the subquery
	s.avg_goals > (SELECT AVG(home_goal + away_goal) 
                    FROM match WHERE season = '2012/2013');


-----------------------------------------------------------------------------------------------------------
----------------------------------------------------CHATPER 3----------------------------------------------
-----------------------------------------------------------------------------------------------------------




-- Chapter 3
-- Section 1
-- Correlated Subqueries

-- special kind of subquery that use values from the outer query to generate results
-- re-run every row generated in the final data set


-- Which match stages tend to have a higher than average number of goals?
--This uqery can be created with a correlated subquery
SELECT
	s.stage,
	ROUND(s.avg_goals, 2) as avg_goal,
	(SELECT AVG(hom_goal + away_goal)
		FROM match
		WHERE season = '2012/2013') AS overall_avg
FROM (SELECT
	stage,
	AVG(home_goal + away_goal) as avg_goals
	FROM match
	WHERE season = '2012/2013'
	GROUP BY stage) as s
WHERE s.avg_goals > (SELECT AVG(home_goal + away_goal)
					FROM match
					WHERE season = '2012/2013'); 

-- Like this
SELECT
	s.stage,
	ROUND(s.avg_goals, 2) AS avg_goal, -- Round off the number of goals by stage
	(SELECT AVG(home_goal + away_goal) -- Select the average for the 2012 season
	FROM match
	WHERE season = '2012/2013') AS overall_avg
FROM
	(SELECT --First Grab the average number of goals by stage of the world
		stage,
		AVG(home_goal+ away_goal) AS avg_goals
	FROM match
	WHRE season = '2012/2013'
	GROUP BY stage) AS s
WHERE s.avg_goals > (SELECT AVG(home_goal + away_goal) --pick the ones with the stage higher than 2012 average.
					FROM match AS m
					WHERE s.stage > m.stage)
;

-- simple and correlated
-- correlated is dependent on the main query to execute 
-- correlated query runs over and over again

SELECT
	c.name AS country,
	AVG(m.home_goal + m.away_goal) AS avg_goals
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
GROUP BY country
;

--What is the average number of goals scored in each country? 
SELECT
	c.name AS country,
	(SELECT AVG(home_goal + away_goal)
	FROM match AS m
	WHERE m.country_id = c.id) 
	AS avg_goals
FROM country AS c
GROUP BY country
;

-- Chapter 3
-- Section 1
-- Exercies

SELECT 
	-- Select country ID, date, home, and away goals from match
    main.country_id,
    main.date,
    main.home_goal, 
    main.away_goal
FROM match AS main
WHERE 
	-- Filter the main query by the subquery
	(home_goal + away_goal) > 
        (SELECT AVG((sub.home_goal + sub.away_goal) * 3)
         FROM match AS sub
         -- Join the main query to the subquery in WHERE
         WHERE main.country_id = sub.country_id)
;


SELECT 
	-- Select country ID, date, home, and away goals from match
	main.country_id,
    main.date,
    main.home_goal,
    main.away_goal
FROM match AS main
WHERE 
	-- Filter for matches with the highest number of goals scored
	(home_goal + away_goal) =
        (SELECT MAX(sub.home_goal + sub.away_goal)
         FROM match AS sub
         WHERE main.country_id = sub.country_id
               AND main.season = sub.season)
;



-- Chapter 3
-- Section 2
-- Nested Subqueries

-- sub queries inside of other subqueries

-- How much did each counry's average differ from the overall average?
SELECT
	c.name AS country,
	AVG(m.home_goal + m.away_goal) AS avg_goals,
	AVG(m.home_goal + m.away_goal) -
		(SELECT AVG(home_goal + away_goal)
		 FROM match) AS avg_diff
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
GROUP BY country
;

--How does each month's total goals differ from the average monthly total of goals scored?
SELECT
	EXTRACT(MONTH FROM date) AS month,
	SUM(m.home_goal + m.away_goal) AS total_goals,
	SUM(m.home_goal + m.away_goal) -
		(SELECT AVG(goals)
			FROM (SELECT
					EXTRACT (MONTH FROM date) AS month,
					SUM(home_goal + away+goal)AS goals
					FROM match
					GROUP BY month)) AS avg_diff
FROM match as m
GROUP BY month
;

-- What is the each country's average goals scored in the 2011/2012 season?
SELECT
	c.name AS country,
	(SELECT AVG(home_goal + away_goal)
	FROM match AS m
	WHERE m.country_id = c.id
		AND id IN (SELECT id
					FROM match
					WHERE season = '2011/2012')) AS avg_goals
FROM country AS c
GROUP BY country
;	


-- Chapter 3
-- Section 2
-- Exercies

SELECT
	-- Select the season and max goals scored in a match
	season,
    MAX(home_goal + away_goal) AS max_goals,
    -- Select the overall max goals scored in a match
   (SELECT MAX(home_goal + away_goal) FROM match) AS overall_max_goals,
   -- Select the max number of goals scored in any match in July
   (SELECT MAX(home_goal + away_goal) 
    FROM match
    WHERE id IN (
          SELECT id FROM match WHERE EXTRACT(MONTH FROM date) = 07)) AS july_max_goals
FROM match
GROUP BY season
;


-- Select matches where a team scored 5+ goals
SELECT
	country_id,
    season,
	id
FROM match
WHERE home_goal >=5 OR away_goal >= 5
;
-- Count match ids
SELECT
    country_id,
    season,
    COUNT(id) AS matches
-- Set up and alias the subquery
FROM 
	(SELECT
    	country_id,
    	season,
    	id
	FROM match
	WHERE home_goal >= 5 OR away_goal >= 5 )
    AS subquery
-- Group by country_id and season
GROUP BY country_id, season
;
SELECT
	c.name AS country,
    -- Calculate the average matches per season
	AVG(outer_s.matches) AS avg_seasonal_high_scores
FROM country AS c
-- Left join outer_s to country
LEFT JOIN (
  SELECT country_id, season,
         COUNT(id) AS matches
  FROM (
      SELECT country_id, season, id
	    FROM match
	    WHERE home_goal >= 5 OR away_goal >= 5) AS inner_s
  -- Close parentheses and alias the subquery
  GROUP BY country_id, season) AS outer_s
ON c.id = outer_s.country_id
GROUP BY country
;




-- Chapter 3
-- Section 3
-- Common Table Expressions 

-- When adding subqueries, complexity increases quickly.
-- CTEs are declared before the main query.
WITH cte AS (
	SELECT col1, col2
	FROM table)
SELECT avg(col1) AS avg_col
FROM cte
;

--EX
SELECT
	c.name AS country,
	COUNT(s.id) AS matches
FROM country AS c
INNER JOIN (
	SELECT country_id, id
	FROM match
	WHERE (home_goal + away_goal) >= 10) AS s
ON c.id = s.country_id
GROUP BY country;


-- Rewrite with a subquery
WITH s AS (
 SELECT country_id, id
 FROM match
 WHERE (home_goal + away_goal) >= 

)
SELECT
	c.name AS country
	COUNT(s.id) AS matches
FROM country AS c
INNER JOIN s
ON c.id = s.country_id
GROUP BY country
;


-- For multiple CTEs just list them with a comma in between, and no comma after the last one.
WITH s1 AS (
	SELECT country_id, id
	FROM match
	WHERE (home_goal + away_goal) >= 10),
WITH s2 AS (
	SELECT country_id, id
	FROM match
	WHERE (home_goal + away_goal) <= 1)
SELECT c.name AS country,
	COUNT(s1.id) AS high_scores,
	COUNT(s2.id) AS low_scores
FROM country AS c
INNER JOIN s1
ON c.id = s1.country_id
INNER JOIN s2
ON c.id = s2.country_id
GROUP BY country
;

-- Why use CTEs? 
-- They are executed only once
-- Better for organization
-- Easy to reference by other queries



-- Chapter 3
-- Section 3
-- Exercies

-- Set up your CTE
WITH match_list AS (
    SELECT 
  		country_id, 
  		id
    FROM match
    WHERE (home_goal + away_goal) >= 10)
-- Select league and count of matches from the CTE
SELECT
    l.name AS league,
    COUNT(match_list.id) AS matches
FROM league AS l
-- Join the CTE to the league table
LEFT JOIN match_list ON l.id = match_list.country_id
GROUP BY l.name
;


-- Set up your CTE
WITH match_list AS (
  -- Select the league, date, home, and away goals
    SELECT 
  		l.name AS league, 
     	date, 
  		m.home_goal, 
  		m.away_goal,
       (m.home_goal + m.away_goal) AS total_goals
    FROM match AS m
    LEFT JOIN league as l ON m.country_id = l.id)
-- Select the league, date, home, and away goals from the CTE
SELECT league, date, home_goal, away_goal
FROM match_list
-- Filter by total goals
WHERE total_goals >= 10
;


-- Set up your CTE
WITH match_list AS (
    SELECT 
  		country_id,
  	  (home_goal + away_goal) AS goals
    FROM match
  	-- Create a list of match IDs to filter data in the CTE
    WHERE id IN (
       SELECT id
       FROM match
       WHERE season = '2013/2014' AND EXTRACT(MONTH FROM date) = 8)
       )
-- Select the league name and average of goals in the CTE
SELECT 
	l.name,
  AVG(match_list.goals)
FROM league AS l
-- Join the CTE onto the league table
LEFT JOIN match_list ON l.id = match_list.country_id
GROUP BY l.name
;


-- Chapter 3
-- Section 4
-- Deciding on techniques to use

-- Differentiating Techniques

-- Joins
	-- Combine 2+ tables
	-- Simple operations/aggregations
	-- Good for multiple tables
	-- "What is the total sales per employee?"

-- Correlated Subqueries
	-- Combine subqueries and tables
	-- avoid limits of joins
	-- Take a long time to process
	-- Good for matching data from different columns in one or more tables
	-- "Who does each employeee report to in a company?"

-- Nested Subqueries
	-- good for multy step transformations
	-- improves accuracy and reproducability
	-- "what is the average deal size closed by each sales representative in the last quarter?"

-- Common table expressions
	-- Organize subqueries
	-- Can Reference other CTEs
	-- comparing a large number of pieces of information
	-- "How did the marketing, sales, growth & engineering teams perform on key metrics?"




-- Chapter 3
-- Section 4
-- Exercises 

SELECT 
	m.id, 
    t.team_long_name AS hometeam
-- Left join team to match
FROM match AS m
LEFT JOIN team as t
ON m.hometeam_id = team_api_id;

SELECT
	m.date,
    -- Get the home and away team names
    hometeam,
    awayteam,
    m.home_goal,
    m.away_goal
FROM match AS m

-- Join the home subquery to the match table
LEFT JOIN (
  SELECT match.id, team.team_long_name AS hometeam
  FROM match
  LEFT JOIN team
  ON match.hometeam_id = team.team_api_id) AS home
ON home.id = m.id

-- Join the away subquery to the match table
LEFT JOIN (
  SELECT match.id, team.team_long_name AS awayteam
  FROM match
  LEFT JOIN team
  -- Get the away team ID in the subquery
  ON match.awayteam_id = team.team_api_id) AS away
ON away.id = m.id
;


SELECT
    m.date,
   (SELECT team_long_name
    FROM team AS t
    -- Connect the team to the match table
    WHERE t.team_api_id = m.hometeam_id) AS hometeam
FROM match AS m;


SELECT
    m.date,
    (SELECT team_long_name
     FROM team AS t
     WHERE t.team_api_id = m.hometeam_id) AS hometeam,
    -- Connect the team to the match table
    (SELECT team_long_name
     FROM team AS t
     WHERE t.team_api_id = m.awayteam_id) AS awayteam,
    -- Select home and away goals
     m.home_goal,
     m.away_goal
FROM match AS m
;


SELECT 
	-- Select match id and team long name
    m.id, 
    t.team_long_name AS hometeam
FROM match AS m
-- Join team to match using team_api_id and hometeam_id
LEFT JOIN team AS t 
ON t.team_api_id = m.hometeam_id
;
-- Declare the home CTE
WITH home AS (
	SELECT m.id, t.team_long_name AS hometeam
	FROM match AS m
	LEFT JOIN team AS t 
	ON m.hometeam_id = t.team_api_id)
-- Select everything from home
SELECT *
FROM home
;

WITH home AS (
  SELECT m.id, m.date, 
  		 t.team_long_name AS hometeam, m.home_goal
  FROM match AS m
  LEFT JOIN team AS t 
  ON m.hometeam_id = t.team_api_id),
-- Declare and set up the away CTE
away AS (
  SELECT m.id, m.date, 
  		 t.team_long_name AS awayteam, m.away_goal
  FROM match AS m
  LEFT JOIN team AS t 
  ON m.awayteam_id = t.team_api_id)
-- Select date, home_goal, and away_goal
SELECT 
	home.date,
    home.hometeam,
    away.awayteam,
    home.home_goal,
    away.away_goal
-- Join away and home on the id column
FROM home
INNER JOIN away
ON home.id = away.id
;



-----------------------------------------------------------------------------------------------------------
----------------------------------------------------CHATPER 4----------------------------------------------
-----------------------------------------------------------------------------------------------------------


-- Chapter 4
-- Section 1
-- Window Functions

-- This will throw an error because non-aggregate values need to be in the group by
SELECT
	country_id,
	season,
	date,
	AVG(home_goal) AS avg_home
FROM match
GROUP BY country_id
;

-- Window Functions!
-- Used to perform calculations on an already generated result set.
-- running totals, rankings and moving averages

--how many goals were scored in each match in 2011/2012 and how did that compage to the average?
SELECT
	date,
	(home_goal + away_goal) AS goals,
	(SELECT AVG(home_goal + away_goal)
		FROM match
	WHERE season = '2011/2012') AS overall_avg
FROM match
WHERE season = '2011/2012'

-- Using a window function
SELECT
	date,
	(home_goal + away_goal) AS goals,
	AVG(home_goal + away_goal) OVER() AS overall_avg
FROM match
WHERE season = '2011/2012'
;

-- Generate a RANK
--What is the rank of matches based on number of goals?
SELECT
	date,
	(home_goal + away_goal) AS goals
FROM match
WHERE season = '2011/2012'
;


SELECT
	date,
	(home_goal + away_goal) AS goals,
	RANK() OVER (ORDER BY home_goal + away_goal DESC) AS goals_rank
FROM match
WHERE season = '2011/2012'
;


-- Chapter 4
-- Section 1
-- Exercies

SELECT 
	-- Select the id, country name, season, home, and away goals
	m.id, 
    c.name AS country, 
    m.season,
	m.home_goal,
	m.away_goal,
    -- Use a window to include the aggregate average in each row
	AVG(m.home_goal + m.away_goal) OVER() AS overall_avg
FROM match AS m
LEFT JOIN country AS c ON m.country_id = c.id
;


SELECT 
	-- Select the league name and average goals scored
	l.name AS league,
    AVG(m.home_goal + m.away_goal) AS avg_goals,
    -- Rank each league according to the average goals
    RANK() OVER(ORDER BY AVG(m.home_goal + m.away_goal)) AS league_rank
FROM league AS l
LEFT JOIN match AS m 
ON l.id = m.country_id
WHERE m.season = '2011/2012'
GROUP BY l.name
-- Order the query by the rank you created
ORDER BY league
;

SELECT 
	-- Select the league name and average goals scored
	l.name AS league,
    AVG(m.home_goal + m.away_goal) AS avg_goals,
    -- Rank leagues in descending order by average goals
    RANK() OVER(ORDER BY AVG(m.home_goal + m.away_goal) DESC) AS league_rank
FROM league AS l
LEFT JOIN match AS m 
ON l.id = m.country_id
WHERE m.season = '2011/2012'
GROUP BY l.name
-- Order the query by the rank you created
ORDER BY league
;


-- Chapter 4
-- Section 2
-- Over with a PARTITION

AVG(home_goal) OVER(PARTITION BY season)

-- How many goals were scored in each match, and how did that compate to the overall average?
SELECT
	date,
	(home_goal + away_goal) AS goals,
	AVG(home_goal + away+goal) OVER() AS overall_avg
FROM match
;

-- How many goals were scored in each match, and how did that compate to the season's average?
SELECT
	date,
	(home_goal + away_goal) AS goals,
	AVG(home_goal + away_goal) OVER(PARTITION BY season) AS Season_avg
FROM match
;

--Partition by multiple columns
SELECT
	c.name,
	m.season,
	(home_goal + away_goal) AS goals,
	AVG(home_goal + away_goal) OVER(PARTITION BY m.season, c.name) AS season_ctry_avg
FROM country as c
LEFT JOIN match AS m
ON c.id = m.country_id

-- PARTITION BY considerations



-- Chapter 4
-- Section 2
-- Exercies
SELECT
	date,
	season,
	home_goal,
	away_goal,
	CASE WHEN hometeam_id = 8673 THEN 'home' 
		 ELSE 'away' END AS warsaw_location,
    -- Calculate the average goals scored partitioned by season
    AVG(home_goal) OVER(PARTITION BY season) AS season_homeavg,
    AVG(away_goal) OVER(PARTITION BY season) AS season_awayavg
FROM match
-- Filter the data set for Legia Warszawa matches only
WHERE 
	hometeam_id = 8673
    OR awayteam_id = 8673
ORDER BY (home_goal + away_goal) DESC
;

SELECT 
	date,
	season,
	home_goal,
	away_goal,
	CASE WHEN hometeam_id = 8673 THEN 'home' 
         ELSE 'away' END AS warsaw_location,
	-- Calculate average goals partitioned by season and month
    AVG(home_goal) OVER(PARTITION BY season, 
         	EXTRACT(MONTH FROM date)) AS season_mo_home,
    AVG(away_goal) OVER(PARTITION BY season, 
            EXTRACT(MONTH FROM date)) AS season_mo_away
FROM match
WHERE 
	hometeam_id = 8673
    OR awayteam_id = 8673
ORDER BY (home_goal + away_goal) DESC
;



-- Chapter 4
-- Section 3
-- Sliding Windows

-- Functions that perform calculations relative to the current row.
ROWS BETWEEN <start> AND <finish>

PRECEDING
FOLLOWING
UNBOUNDED PRECEDING
UNBOUNDED FOLLOWING
CURRENT ROW

-- Manchester City Home Games
SELECT
	date,
	home_goal,
	away_goal,
	SUM(home_goal)
		OVER(ORDER BY date ROWS BETWEEN
				UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM match
WHERE hometeam_id = 8456 AND season = '2011/2012'
;

SELECT date,
	home_goal,
	away_goal,
	SUM(home_goal)
		OVER(ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS last2
FROM match
WHERE hometeam_id = 8456
	AND season = '2011/2012'
;
-- Chapter 4
-- Section 3
-- Exercies

SELECT 
	date,
	home_goal,
	away_goal,
    -- Create a running total and running average of home goals
    SUM(home_goal) OVER(ORDER BY date 
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
    AVG(home_goal) OVER(ORDER BY date
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_avg
FROM match
WHERE 
	hometeam_id = 9908 
	AND season = '2011/2012';

SELECT 
	-- Select the date, home goal, and away goals
	date,
    home_goal,
    away_goal,
    -- Create a running total and running average of home goals
    SUM(home_goal) OVER(ORDER BY date DESC
         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS running_total,
    AVG(home_goal) OVER(ORDER BY date DESC
         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS running_avg
FROM match
WHERE 
	awayteam_id = 9908 
    AND season = '2011/2012';


-- Chapter 4
-- Section 4
-- Bringing it all together

-- Who beat Manchester United?


-- Chapter 4
-- Section 4
-- Exercises

SELECT 
	m.id, 
    t.team_long_name,
    -- Identify matches as home/away wins or ties
	CASE WHEN m.home_goal > m.away_goal THEN 'MU Win'
		WHEN m.home_goal <  m.away_goal THEN 'MU Loss'
        ELSE 'Tie' END AS outcome
FROM match AS m
-- Left join team on the home team ID and team API id
LEFT JOIN team AS t 
ON m.hometeam_id = t.team_api_id
WHERE 
	-- Filter for 2014/2015 and Manchester United as the home team
	m.season = '2014/2015'
	AND t.team_long_name = 'Manchester United';



SELECT 
	m.id, 
    t.team_long_name,
    -- Identify matches as home/away wins or ties
	CASE WHEN m.home_goal > m.away_goal THEN 'MU Loss'
		WHEN m.home_goal < m.away_goal THEN 'MU Win'
        ELSE 'Tie' END AS outcome
-- Join team table to the match table
FROM match AS m
LEFT JOIN team AS t 
ON m.awayteam_id = t.team_api_id
WHERE 
	-- Filter for 2014/2015 and Manchester United as the away team
	season = '2014/2015'
	AND t.team_long_name = 'Manchester United'
;


-- Set up the home team CTE
WITH home AS (
  SELECT m.id, t.team_long_name,
	  CASE WHEN m.home_goal > m.away_goal THEN 'MU Win'
		   WHEN m.home_goal < m.away_goal THEN 'MU Loss' 
  		   ELSE 'Tie' END AS outcome
  FROM match AS m
  LEFT JOIN team AS t ON m.hometeam_id = t.team_api_id),
-- Set up the away team CTE
away AS (
  SELECT m.id, t.team_long_name,
	  CASE WHEN m.home_goal > m.away_goal THEN 'MU Win'
		   WHEN m.home_goal < m.away_goal THEN 'MU Loss' 
  		   ELSE 'Tie' END AS outcome
  FROM match AS m
  LEFT JOIN team AS t ON m.awayteam_id = t.team_api_id)
-- Select team names, the date and goals
SELECT DISTINCT
    m.date,
    home.team_long_name AS home_team,
    away.team_long_name AS away_team,
    m.home_goal,
    m.away_goal
-- Join the CTEs onto the match table
FROM match AS m
LEFT JOIN home ON m.id = home.id
LEFT JOIN away ON m.id = away.id
WHERE m.season = '2014/2015'
      AND (home.team_long_name = 'Manchester United' 
           OR away.team_long_name = 'Manchester United');


-- Set up the home team CTE
WITH home AS (
  SELECT m.id, t.team_long_name,
	  CASE WHEN m.home_goal > m.away_goal THEN 'MU Win'
		   WHEN m.home_goal < m.away_goal THEN 'MU Loss' 
  		   ELSE 'Tie' END AS outcome
  FROM match AS m
  LEFT JOIN team AS t ON m.hometeam_id = t.team_api_id),
-- Set up the away team CTE
away AS (
  SELECT m.id, t.team_long_name,
	  CASE WHEN m.home_goal > m.away_goal THEN 'MU Loss'
		   WHEN m.home_goal < m.away_goal THEN 'MU Win' 
  		   ELSE 'Tie' END AS outcome
  FROM match AS m
  LEFT JOIN team AS t ON m.awayteam_id = t.team_api_id)
-- Select columns and and rank the matches by goal difference
SELECT DISTINCT
    m.date,
    home.team_long_name AS home_team,
    away.team_long_name AS away_team,
    m.home_goal, m.away_goal,
    RANK() OVER(ORDER BY ABS(home_goal - away_goal) DESC) as match_rank
-- Join the CTEs onto the match table
FROM match AS m
LEFT JOIN home ON m.id = home.id
LEFT JOIN away ON m.id = away.id
WHERE m.season = '2014/2015'
      AND ((home.team_long_name = 'Manchester United' AND home.outcome = 'MU Loss')
      OR (away.team_long_name = 'Manchester United' AND away.outcome = 'MU Loss'));