// 1. the number of deleted messages in the dataset
db.tweets.find(
{ "delete": {$exists: true} }
).count()

// 2. the number of tweets that are replies to another tweet
db.tweets.find(
{ "in_reply_to_status_id": {$ne: null}}
).count()

// 3. the five user IDs (field name: uid) that have tweeted the most
db.tweets.aggregate([
{ $match: {"user.id": {$ne: null}} },
{
	$group:{
		_id: "$user.id_str",
		count: {$sum: 1}
	}		
},
{ $sort: {count: -1} },
{ $limit: 5}
])


