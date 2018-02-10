import twitter_pb2 as pb2
#import sys

# main code
twts = pb2.Tweets()
with open("twitter.pb", "rb") as file:
	twts.ParseFromString(file.read())
	file.close()

# 1. Find the number of deleted messages in the dataset
# 2. Find the number of tweets that are replies to another tweet
# 3. Find the five user IDs (field name: uid) that have tweeted the most
count_del = 0
count_re, dedup  = 0, dict()
count_uid = dict()
for twt in twts.tweets:
	if twt.is_delete: count_del += 1
	if twt.HasField("insert") and twt.insert.reply_to and str(twt.insert.id) not in dedup: 
		dedup[str(twt.insert.id)] = ""
		count_re += 1
	if twt.HasField("insert"):
		if str(twt.insert.uid) not in count_uid:
			count_uid[str(twt.insert.uid)] = 1
		else:
			count_uid[str(twt.insert.uid)] += 1

count_uid = sorted(count_uid.items(), key=lambda x:x[1], reverse=True)
count_uid = [str(uid) for uid, count in count_uid[0:5]]

print("Number of Deleted Messages: %d" % count_del)
print("Number of tweets that are replies to another tweet: %d" % count_re)
print("UID of the top 5 users who have tweeted the most: \n%s" % "\n".join(count_uid))
