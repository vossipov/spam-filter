Content:  
* [Description](#description)  
* [Bayes Theorem](#what-is-bayes-theorem)  
* [TFIDF](#tfidf)
* [Install](#install)
## Prerequisites
* [Python 3.7](https://www.python.org/downloads/release/python-377/)
* [NumPy](https://numpy.org/)
* [nltk](https://www.nltk.org/)
* [pandas](https://pandas.pydata.org/)

## Description
Soon...

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

## TFIDF
**TFIDF**, short for term *frequency–inverse document frequency*  is a numerical statistic that is intended to reflect 
how important a word is to a document in a collection or corpus

## Install
    $ pip install -r requirements.txt
    
or
    
    $ python setup.py install
## References
Soon...