# Turkish Diacritisation | YZV 405E NLP Term Project

Author: Bora Boyacıoğlu

Student ID: 150200310

This project involves *ASCII-fying* of Turkish texts by using a Natural Language model and rule based algorithms. It is split into 5 sections.

## Datasets

The main data source is from the project topic itself. There is a train and a test data, which include diacritised and undiacritised sentences, respectively. In both of the datasets, there are `ID` and `Sentence` columns.

Also, there is a vocabulary, used in the rule based part *(Section 4)* by Kemal Ogün Işık, [GitHub @ guncel-turkce-sozluk](https://github.com/ogun/guncel-turkce-sozluk), 2021.

### Test Data Example
|ID|Sentence|
|-|-|
|18|diyalog icin de kapi acildi aclik grevle...|
|24|Israil cocuklara acimiyor Erdogan Kah...|
|34|kadini cahil tutmak suretiyle somurme...|
|55|isim secimimin otobiyografik nedenle...|

### Train Data Example
|ID|Sentence|
|-|-|
|25|isterseniz hızlı arama veya detaylı ara...|
|46|bu iki yemek tarifimiz girit yemekleri...|
|66|krep hamuru bitene kadar i ̧sleme deva...|
|99|ilk önce hayatsal fonksiyonları neden...|

### Vocabulary Example
|ID|madde|
|-|-|
|1084|ünlü|
|3408|dağ|
|4958|doğululaşma|
|11505|seyrekleştirmek|

## Sections

### 1. Preprocessing

In this step, I processed the train and test data. The steps involved are:

---
* **1.1.a. Normalisation:** Characters lowered and extra (non-alphabetical) chars removed.
* **1.1.b. Tokenisation:** Using NLP, the text is tokenised.
---
* **1.2.a. Vocabulary Creation:** Combining both of the train and test data, a vocabulary is created with the indexes and words.
* **1.2.b. Padding:** The sentences are padded into a stable length.
* **1.2.c. Converting to Indices.** Using the vocabulary, padded sentences are converted into indices.
---

### 2. Training

This step involves writing an **Encoder-Decoder Seq2Seq** model and training it with the preprocessed train data. My model includes an embedding, an LSTM, and a dropout layer for the encoder. And the decoder also includes a linear layer. The Seq2Seq part combines these in its forward pass.

I used the following parameters for the training:

```
embedding_dimension = 64
hidden_dimension = 256
n_layers = 2
dropout = 18
batch_size = 18
num_epochs = 50
```

With the **Adam** optimiser and **CrossEntropyLoss**. And got the train loss at $3.89$. The training loop includes early stopping, as well as KeyboardInterrupt stopping. In each epoch, the model is saved as long as it has a lower loss than the previous epoch. This is also important keeping in mind that my model is actually computationally heavy. Using Google Colab's A100 GPU, it took 32 minutes on average for each epoch to train. This is more than **25 hours of training** in total. And not even including the failed attempts.

### 3. Evaluation

I started the evaluation by first importing the model and the data. The vocabulary and the test data is used here. Using the model's decoder, I predicted all the sentences in the test data.

The scores here, compared to the gold data, are calculated using the two scores: **Word Score** (Word-Level Accuracy) and **Sentence Score** (Sentence-Level Subset Accuracy). This way, my word score is $2.60\%$, and the sentence score is $0.52\%$. Acknowledging the scores being unimagenably low, I continued with the next steps.

### 4. Rule Based Algorithms

Here, I have listed three cases and prepared my data for them.

---
* **Case 1:** The word doesn't need to be changed. It is already in the ASCII form. That way, it is possible to simply put it as it is to the output sentence.
* **Case 2:** According to my vocabularies, there is only one possible acronym for that word. Then, put that version to the output.
* **Case 3:** If there multiple acronyms to chose from, ask the model. If one of the acronyms is in the prediction, use it. Otherwise, select the most probable acronym according to the occurrence in the train data.
---

To use these cases, I created a vocabulary using both the train data and the **Turkish Vocabulary** data. I made a set of all the words I have, which is $150286$. Then, I listed the ones that has non-ASCII chars (Case 2), which is $85486$. This already eliminated the $43.12\%$ of the words to stay in the Case 1. Later, I have listed the acronyms that correspond to words with more than one acronym, which is $1789$. And I found out that at the end, only $1.19\%$ of the whole vocabulary need to go into Case 3.

### 5. Combination of the Model and the Rule Based Algorithms

Using the predictions from the model and the cases from the rule based algorithms, I implemented a selection method. Using this method, I finally got the results. With my Word Score and Sentence Score calculations, I get the accuracies $81.88\%$ as Word Score and $14.01\%$ as Sentence Score.

### Comments

I truly acknowledge that my scores are much less than what they should have been. This is largely due to the structure of the model. My model utilises the sentences as a whole, instead of words or even letters. That could've boosted the performance up to 2-digit numbers. Also, there should've been more rules. With some other easy implementations, the words may have been predicted better.