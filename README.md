# Quora-Duplicate-Challenge
In this project, through advanced feature engineering and preprocessing of data an nlp model is trained to solve the Quora-Duplicate-Challenge posted on Kaggle a couple of years ago found at https://www.kaggle.com/c/quora-question-pairs

The data is big so I used only 30000 lines in the start and then for final advanced feature engineering I used 120000 lines still not the whole dataset and got incredible results! So Let's go through this journey and see the processes I went through to approach this problem. 

-------------------To see the final result please scroll all the way down there I have screenshots of the Heroku website attached.-------------------------------------

![image](https://user-images.githubusercontent.com/54567394/170813278-421c4436-f6e6-4160-a850-8f4210eacac5.png)

I first tried checking only bag of words and then through CountVectorizer and then splitting the data I found through RandomForestClassifier that the accuracy with just the given data is 74.02% and through XGBClassifier the accuracy was about 72.45 %

Now that I knew that just bow (bag of words) can lead us to about 72 - 74%, I tried adding additional basic features. I then tried adding features such as - 
q1_len -> char length of q1
q2_len -> char length of q2
q1_words -> #words in q1
q2_words -> #words in q2
words_common -> # of common unique words
words_total -> total # of words in q1 + total # of words in q2
word_share -> words_common / words_total

Meaning for every row we would have to make 7 new rows which would be a total of 6007 features, 3000 from q1 + 3000 from q2 and then the additional 7 we add. 

Before doing any of this work we check the distribution of duplicate and non-duplicate questions in our model and obtain the plot below. 

![image](https://user-images.githubusercontent.com/54567394/170847345-c3a91b97-2026-4a5d-a472-a45930cbb7c7.png)

And then I also checked the repeated question histogram to make more sense of the dataset and obtained the following graph

![image](https://user-images.githubusercontent.com/54567394/170847379-4a48ed29-1427-4665-90ef-4ead9196e11c.png)

So this tells me a lot about what is in my data set and i can start applying the further feature engineering algorithms. NOTE - we have not done any preprocessing on this data at this time.


I won't go over the code in readme but it is all commented well in the file 'bagofwords_with_baic_7features.ipynb'


After inclusing all the features, I wanted to analyse the features to see if they are giving us some important information about the data. 

Analysing q1 - ![image](https://user-images.githubusercontent.com/54567394/170847451-6aba95fc-7703-408d-8033-f58198501aac.png)

Analysing q2 - ![image](https://user-images.githubusercontent.com/54567394/170847457-c8edec4a-9b84-4b3a-87a2-d9f2f8f5bfab.png)

Analyzing q1_num_words - ![image](https://user-images.githubusercontent.com/54567394/170847472-f83975ec-057c-43ab-9b36-33c2fc8f59eb.png)

Analyzing q2_num_words - ![image](https://user-images.githubusercontent.com/54567394/170847483-524962af-fd29-44cb-83a2-b9ab056ddda6.png)

Analysing common words and total words - ![image](https://user-images.githubusercontent.com/54567394/170847496-9e32e22a-5cfe-487d-bff5-8891571d15a8.png)

Analyse word share - ![image](https://user-images.githubusercontent.com/54567394/170847508-a435072a-e79b-4ffb-8d94-2d29f9420512.png)

From these graphs we get some crucial information. In words share - if the wordshare is less than 0.2 then theres more odds of it being non duplicate and if the words_share is more than 0.2 then there is more chance in it being duplicate. Similar trends are seen in words_common where less than 5 common words is non-duplicate and more than 5 is duplicate. 

I then proceed on doing bagofwords on this new ques_dataframe and after concatenating the columns and doing RandomForest we find the accuracy to be 76.8%. and XGBoost gave 76.45%

Meaning after adding these 7 features, we have gotten to atleast a 2% increase in accuracy. 
NOTEE - we are not even using the whole dataset due to RAM issues (we have not done any deep learning at this point)

Now I thought about the further feature we can implement to get this Problem to be solved with higher accuracy. 

So now we have to think about what additional Features can be added into this.
In total I divided these addional features into 3 SEPARATE sections - I'll cover token features first and then go through Length based features and then finally discuss the Fuzze features.

Coming onto the Token features first I thought about 8 features in this category that can be added

1. cwc_min -> Ratio of number of common words to the length of smaller question between q1 and q2
2. cwc_max ->  Ratio of number of common words to the length of larger question between q1 and q2
3. csc_min -> Ratio of number of common stopwords to the less number of stopwords in the 2 questions
4. csc_max -> Ratio of number of common stopwords to the more number of stopwords in the 2 questions
5. ctc_min -> Ratio of number of common tokens to the minimum number of token in the 2 questions
6. ctc_max -> Ratio of number of common tokens to the maximum number of token in the 2 questions
7. last_word_eq -> is the last word equal?
8. first_word_eq -> is the first word equal?

Now lets focus on some Length based fetaures - 

1. mean_length -> mean of the length of the 2 questions
2. abs_len_diff -> absolute value of (tokens of q1 - tokens of q2)
3. longest_substring_ratio -> the longest continuous substring that can be found in both sentences. 

Fuzzy Features - They are honeslty a lot but can be found on https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

v cool btw ^^ and since the library fuzzywuzzy already has them we dont have to personaly implement them

Now lets talk about some preprocessing that can help our model get even better. Bascically in the Preprocessing I did, I converted al the symbols to their respective words such as '%' to 'percent' '@' to 'at' etc. I also replaced any mathematical symbol to just blank space. From wikipedia, I found a list of contractions and their decontractions so I decontracted everything in the text and got a simple english lower case version of the whole dataset. well 120000 lines :)

You can look at the code to look at how I implemented all the features but I will go over some analyses here. 

So looking at our min features in the token_features we get ![image](https://user-images.githubusercontent.com/54567394/170847959-8b8eca31-cc43-4a49-bde7-60a7e15f7ab0.png)

looking at max - ![image](https://user-images.githubusercontent.com/54567394/170847964-7d7717fd-65a6-4f99-b63f-7e88867ad5e7.png)

From these 2 (looking at the diagonal graphs) I can tell that there is a point in the graphs where there is a distinction between duplicate or not so these are some helpful features.

![image](https://user-images.githubusercontent.com/54567394/170847982-ef5369be-d75f-4a9b-975b-e1544dadf871.png)
First word and last work features are also helpful

Analysing the fetaures in the 'Length_based_feature' category we get this graph. 

![image](https://user-images.githubusercontent.com/54567394/170847995-1cb13448-f71e-4104-a15f-803fbc5e1520.png)
Although abs_legth_Diff and mean_length might not seem to be too helpful I still kept them in the data  - They could also have been taken out but I have decided to keep them.

Our fuzzy features proved out to be very useful giving the following graph. 
![image](https://user-images.githubusercontent.com/54567394/170848015-e8c05754-be85-46ee-b55a-ef0a3bd8f8a6.png)

Now doing our usual testing through RandomForest and XGBoost we get an accuracy of 80.88 and 80.42% respectufully!!! That is amazing considereing we have only used 25% of the dataset and not implemented deep learning AT ALL. 

The next step was to putting this all into a heroku deployed website so everyone can test it!

we needed to get the questions preprocessed into the rf.predict() function and then after the input display it back. Honestly a loing process but learning about Heroku CLI and pycharm webdev was very fun. 

You can look at the source code linked in this post and the final version came out to be like this! -

![image](https://user-images.githubusercontent.com/54567394/170848131-7169cdfa-e6a1-4850-abe5-0cebb252378c.png)

 Now lets talk about how we can improve this thing even further and also what happens if it predicts wrong. We never want it to predict a not-duplicate question duplicate but it is stilll okay if the duplicate question is not-duplicate since it saves rthe company a lot of hassle. Anyways to get better one obvious thing is to increase the data (Use all of the data)
 We can add more features (we only have 22 rn)
 Instead of only trying Bag_of_Words try tfidf or word2vec or even tfidf weighted word2vec. 
 IN preprocessing inculde stemming and more techniques., 
 Apply more algorithms -We only used RandomForest and XGBoost, we can also use SVM, Logistic Regression, Hyperparameter tuning etc.)
 
Implementing deep learning and BERT transformer can also help as you can upload as much data as you want and by putting it onto the cloud you dont have to worry about your own RAM.
Or we can send small batches of data and apply incremental learning.

This all would increase the accurtacy of our project!!

I really enjoyed this challenge, I hope this readme was helpful.
 
 

