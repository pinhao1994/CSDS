-- 1. the number of deleted messages in the dataset
select count('is_delete') from tweets where is_delete = 1;

-- 2. the number of tweets that are replies to another tweet 
-- select count('reply_to') from tweets where reply_to != 0;
select count(distinct id) from tweets where reply_to != 0;

-- 3. the five user uid that have tweeted the most
select uid from tweets where text != "" group by uid order by count(distinct id) DESC limit 5;

