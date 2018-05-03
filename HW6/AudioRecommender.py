
# coding: utf-8

# # Audio Recommender
# ### Pin-Hao Chen (phc2121) | Sharon Tsao (sjt2141)

# In[1]:


from pyspark.mllib.recommendation import *
from operator import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("sparkhw-csds").getOrCreate()
sc = spark.sparkContext


# In[2]:


rawUserArtistData = sc.textFile("s3n://sparkhw-csds/user_artist_data.txt")
rawArtistData = sc.textFile("s3n://sparkhw-csds/artist_data.txt")
rawArtistAlias = sc.textFile("s3n://sparkhw-csds/artist_alias.txt")


# In[3]:


def helper1(line):
    pairs = line.split("\t")
    return (pairs[0], pairs[1].strip()) if len(pairs) == 2 else (None, None)

artistByID = rawArtistData.flatMap(lambda l: (helper1(l), ))
artist_df = artistByID.toDF(['artistid', 'name'])
artist_df = artist_df.filter(artist_df.artistid != "null")
# artist_df.show(5)


# In[4]:


def helper2(line):
    tokens = line.split("\t")  
    return (tokens[0], tokens[1].strip()) if tokens[0] else (None, None)   
    
bad_good = rawArtistAlias.flatMap(lambda l: (helper2(l), ))
artistAlias = bad_good.collectAsMap()
artistAlias.pop(None, None)
# artist_alias_df = bad_good.toDF(['badid', 'goodid'])
# artist_alias_df = artist_alias_df.filter(artist_alias_df.badid != "null")
# artist_alias_df.show(5)


# In[5]:


def helper3(line):
    userID, artistID, count = line.split(" ")
    finalArtistID = artistID
    if artistID in bArtistAlias.value:
        finalArtistID = bArtistAlias.value[artistID]
    return Rating(int(userID), int(finalArtistID), int(count))

bArtistAlias = sc.broadcast(artistAlias)
trainData = rawUserArtistData.map(lambda l: helper3(l)).cache()
#userArtist_df = trainData.toDF(["userid", "artistid", "playcount"])


# In[6]:


model = ALS.trainImplicit(trainData, 10, iterations=5, lambda_=0.01, blocks=1)


# In[7]:


target_user = 2093760
recommendations = model.recommendProducts(target_user, 10)


# In[8]:


#recommendations


# In[9]:


recommendedProductIDs = set(map(lambda r: r.product, recommendations))
#recommendedProductIDs


# In[10]:


rmd_lt = set(map(lambda e: e.name, artist_df.filter(artist_df.artistid.isin(recommendedProductIDs)).collect()))
print(rmd_lt)

