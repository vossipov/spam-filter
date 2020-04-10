### Content Table:
- [Prerequisites](#prerequisites)  
- [Introduction](#introduction)  
- [Bayes Theorem](#what-is-bayes-theorem)  
- [Naive Bayes Classifier](#naive-bayes-classifier)
    * [Problem statement](#problem-statement)
        * [Probabilistic model](#probabilistic-model)
        * [Constructing a classifier](#constructing-a-classifier)
* [Install](#install)
## Prerequisites
* [Python 3.7](https://www.python.org/downloads/release/python-377/)
* [NumPy](https://numpy.org/)
* [nltk](https://www.nltk.org/)
* [pandas](https://pandas.pydata.org/)

## Introduction
Spam is information crafted to be delivered to a large number of recipients, 
in spite of their wishes. A spam filter is an automated 
tool to recognize spam so as to prevent its delivery. 
The purposes of spam and spam filters are diametrically opposed: 
spam is effective if it evades filters, 
while a filter is effective if it recognizes spam. 
The circular nature of these definitions, 
along with their appeal to the intent of sender 
and recipient make them difficult to formalize. 
A typical email user has a working definition no more 
formal than "I know it when I see it." Yet, current 
spam filters are remarkably effective, more effective 
than might be expected given the level of uncertainty and debate over a formal definition of spam, more effective than might be expected given the state-of-the-art information retrieval and machine learning methods for seemingly similar problems.

## What is Bayes Theorem?
Bayes' Theorem is a simple mathematical formula used for calculating conditional probabilities. 
It figures prominently in subjectivist or Bayesian approaches to epistemology, statistics, and inductive logic. 
Subjectivists, who maintain that rational belief is governed by the laws of probability, lean heavily on conditional 
probabilities in their theories of evidence and their models of empirical learning. Bayes' Theorem is central to these 
enterprises both because it simplifies the calculation of conditional probabilities and because it clarifies significant
features of subjectivist position.  

It is mathematically expressed as:   
<img src="https://render.githubusercontent.com/render/math?math=P(A|B) = \frac{P(B|A)P(A)}{P(B)}">
<br>
where <img src="https://render.githubusercontent.com/render/math?math=A"> 
and 
<img src="https://render.githubusercontent.com/render/math?math=B"> are events and <img src="https://render.githubusercontent.com/render/math?math=P(B)\ne 0">
* <img src="https://render.githubusercontent.com/render/math?math=P(A | B)"> the likelihood of event <img src="https://render.githubusercontent.com/render/math?math=A"> occurring given <img src="https://render.githubusercontent.com/render/math?math=B"> is true
* <img src="https://render.githubusercontent.com/render/math?math=P(B | A)"> the likelihood of event <img src="https://render.githubusercontent.com/render/math?math=B"> occurring given <img src="https://render.githubusercontent.com/render/math?math=A"> is true
* <img src="https://render.githubusercontent.com/render/math?math=P(A)"> and <img src="https://render.githubusercontent.com/render/math?math=P(B)"> are the probabilities of observing  <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=B"> respectively
## Naive Bayes Classifier
_Naive Bayes Classifier_ is a simple and powerful classification algorithm based on [Bayes' theorem](#what-is-bayes-theorem) with naive independence assumptions between the feature.   
The classifier tries to choose the most probable class based on what it has learned about the features(presence or frequency of words in the emails of each type)
<br>
### Problem statement
#### Probabilistic model
Let we have <img src="https://render.githubusercontent.com/render/math?math=C = \{C_{1},C_{2},...,C_{k}\}"> where <img src="https://render.githubusercontent.com/render/math?math=C"> set of unique classes and <img src="https://render.githubusercontent.com/render/math?math=M=\{w_{1},w_{2},..., w_{n}\}"> where <img src="https://render.githubusercontent.com/render/math?math=M">
set of unique words in message, so now we want to determine which class the message belong to.
<br>According to [Bayes' Theorem](#what-is-bayes-theorem) we got following:

<img src="https://render.githubusercontent.com/render/math?math=P(C_{k} | M)= \frac{P(C_{k})P(M|C_{k})}{P(M)}">

<br>In practice, there is interest only in the numerator of that fraction, because the denominator does not depend on 
<img src="https://render.githubusercontent.com/render/math?math=C_{k}"> 
and the values of the features <img src="https://render.githubusercontent.com/render/math?math=M"> are given, so that the denominator is effectively constant. 
The numerator is equivalent to the joint probability model <img src="https://render.githubusercontent.com/render/math?math=P(C_{k}|w_{1},w_{2},w_{3},...,w_{n})"> 
<br>which is can be rewritten as follows:<br>
<img src="https://render.githubusercontent.com/render/math?math=P(C_{k}|w_{1},w_{2},w_{3},...,w_{n}) = P(w_{1} \cap w_{2} \cap w_{3} \cap ... \cap w_{n}| C_{k})P(C_{K})"> 
<br>now using the _naive conditional independence_ we got: <br>

<img src="https://render.githubusercontent.com/render/math?math=P(C_{k}|w_{1},w_{2},w_{3},...,w_{n}) = P(w_{1}|C_{k})P(w_{2}|C_{k})P(C_{k}| w_{3})...P(w_{n}| C_{k})P(C_{K})"> 

#### Constructing a classifier.
After all of the above discussions we have got the naive Bayes probability model. The Naive Bayes Classifier combines this probability model and decision rule(function which maps an observation to an appropriate action).
<br>The most common rule is to pick the hypothesis that is most probable, also it know as the _maximum a posterior_. Now we can describe the classifier model as follows:
<br>

   <img src="https://render.githubusercontent.com/render/math?math=\hat{c} = argmaxP(C_{k})\prod_{i = 1}^{n}{P(w_{i}|C_{k})}"> 

## Implementation
 Soon...
## Install
    $ pip install -r requirements.txt
    
or
    
    $ python setup.py install
## References
[[1]](https://www.aclweb.org/anthology/Q19-1004.pdf) Belinkov, Y. and Glass, J. (2019). Analysis Methods in Neural Language Processing: A Survey.Transactions of the Association for Computational Linguistics<br>
[[2]](https://books.google.ru/books/about/Email_Spam_Filtering.html?id=h6AYzY-yWZ8C&redir_esc=y) Cormack, G. (2008). Email Spam Filtering: A Systematic Review.Foundations and Trends® in Information Retrieval
