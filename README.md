# Quora-Duplicate-Challenge
In this project, through advanced feature engineering and preprocessing of data an nlp model is trained to solve the Quora-Duplicate-Challenge posted on Kaggle a couple of years ago found at https://www.kaggle.com/c/quora-question-pairs

The data is big so i used only 30000 lines in the start and then for final advanced feature engineering I used 120000 lines still not the whole dataset and got incredible results! So Let's go through this journey and see the processes I went through to approach this problem. 

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


