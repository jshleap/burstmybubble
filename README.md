# burstmybubble
We live in an increasignly polarized world, and getting balanced information from news media can be challenging, making most of us live in a political bubble and that includes editors reviewing the news, and Public Relation professionals creating statements. 

In therms of news, most people tend to judge the outlet as a whole instead of weighing-in each article. This app allows you to rate individual news and get a sense of bias of the content you are reading. 

## Getting the political polarity
In this project I am using the [convote v1.1](http://www.cs.cornell.edu/home/llee/data/convote.html) from Cornell university, especifically data_stage_two, as a training set. Briefly, this dataset contains downloaded all available pages all records from the 2005 U.S. House floor debates from [govtrack.us](govtrack.us), that has been labeled based on party, bill and speaker (for more details check [here](http://www.cs.cornell.edu/home/llee/data/convote/README.v1.1.txt)). This data is then processed through NLP using the ngram Bag of Words (BOW) strategy, and then estimate a discriminant function based on the labeled training data.

## Testing news
News will be scraped from the web and input to the NLP the same way that the training data was. Both the training data and the incoming news will be aligned and the BOW vector from the news will be used as input for the discriminant function that in turns will predict the probabilities of the sample to belong to either political leaning.

### Ideal case (i.e. Not this project)
The ideal case for testing this app will be to have a manually labelled set of news on the same topic as left, right, or neutral. Ideally, this will be done with randomly selected raters with diverse backgrounds to remove subjectivity from the labelling. However, for this project I will select the news myself, and therefore some bias might remain.

