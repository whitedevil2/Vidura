#!/usr/bin/env python
# coding: utf-8

# ## Efficient Q&A and search system for Parliamentary Questions
# 
# ### Problem Statement: 
# 
#  *During parliament session, each Department receives number of Parliament Questions on varied topics raised by MPs and handled in a very time bound manner on top priority. Each reply is generally prepared by seeking inputs from all the other relevant departments which requires lot of efforts and is also time consuming. It is desired a platform can be designed which can provide responses to similar PQ asked earlier, suggest probable reply and indicate different departments having similar programs and information. This will be helpful in preparing proper reply to PQ. As of now there are some search tools, separate for Lok Sabha and Rajya Sabha. But a unified, fast and effective mechanism is missing.*

# In[25]:


#Importing libraries
import sys
import spacy
import pandas as pd
import math
import re
import numpy as np
from collections import Counter
import gzip
import gensim 
import logging
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import json

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
from cleantext import clean


# In[5]:


#Fetching the data
data=pd.read_csv("sample_data/rajyasabha_questions_and_answers_2009.csv")
questions=data['question_description']
answers=data['answer']
titles=data['question_title']


# ## Step 1: Creating a cosine similarity calculator
# 
# This helps in calculating similarity between two sentences. Since we are planning to take a) Title b) Question as input from the user, using the 'Title' we will first try to get the relevant rows to search through.
# 
# ![title](img/step1.png)

# In[6]:


#WORD NET MODULES
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

def get_best_synset_pair(word_1, word_2):
    
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim!=None and sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair
    
def length_dist(synset_1, synset_2):
    
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))

def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))

def most_similar_word(word, word_set):
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim

brown_freqs = dict()
N=0
def info_content(lookup_word):
    
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))

def semantic_vector(words, joint_words, info_content_norm):
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec       


# In[7]:


#Function to find similarity between two strings that are converted into vectors
WORD = re.compile(r"\w+")

def get_cosine(sentence_1, sentence_2):
    sentence_1 = re.sub('[^A-Za-z0-9\s]', '', sentence_1).lower()
    sentence_2 = re.sub('[^A-Za-z0-9\s]', '', sentence_2).lower()
    info_content_norm = True
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


# In[8]:


#Function to get top 100 relevant rows to search
def get_relevant_indices(query,title_list):
    
    cosine_list=np.empty(len(title_list))
    for i in range(len(title_list)):
        cosine_val_i=get_cosine(query,title_list[i])
        cosine_list[i]=cosine_val_i
    
    relevant_indices=cosine_list.argsort()[-10:][::-1]
    return relevant_indices


# In[9]:


#Function to get relevant titles
def get_relevant_titles(query):
    titles=data['question_title'].values.tolist()
    relevant_indices=get_relevant_indices(query,titles)
    relevant_titles=[]
    for i in range(len(relevant_indices)):
        relevant_titles.append(titles[relevant_indices[i]])
    
    relevant_titles_df=pd.DataFrame({'Indices':relevant_indices,
                                    'Titles':relevant_titles})
    return relevant_titles_df


# In[16]:





# In[21]:


#paragraph to search for the answer
#def get_paragraph(relevant_titles):
  #paragraph=''
  #for i in relevant_titles['Indices'].values:
    #paragraph+=answers[i]+'\n'
  #return paragraph

#paragraph=get_paragraph(relevant_titles)

#print(paragraph)


# In[18]:


#Importing the BERT pretrained on SQuAD

#Model
#model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

#Tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[19]:


#Answer prediction function
#def get_answer(query,paragraph):
    
    #Don't worry much about these lines. These will just convert the input
    #into a form that the BERT model can accept
    
    #encoding = tokenizer.encode_plus(text=query,text_pair=paragraph,max_length=512,truncation=True)
    #inputs = encoding['input_ids']  #Token embeddings
    #sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    #tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
    
    #Feeding the input into the model
    #start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
    
    #Bert assigns scores to each token. The tokens that have the highest
    #Start and end scores are likely to be our starting and ending of the
    #answer. Below lines of code will do that. For better understanding, refer
    #the functioning of BERT online
    
    #start_index = torch.argmax(start_scores)
    #end_index = torch.argmax(end_scores)
    #answer = ' '.join(tokens[start_index:end_index+1])
    
    #BERT uses word peice tokenization. i.e. playing = play##ing. To make our
    #output look neat, let's removes those ##
    
    #corrected_answer = ''
    #for word in answer.split():
        #If it's a subword token
        #if word[0:2] == '##':
            #corrected_answer += word[2:]
        #else:
            #corrected_answer += ' ' + word
    
    #return corrected_answer


# In[20]:





# In[10]:


#main function
def main():
  title=sys.argv[1]
  #question=sys.argv[2]
  #fetching relevant titles
  relevant_titles=get_relevant_titles(title)

  #searching the paragraph for a suitable answer using bert
  #answer=get_answer(question,paragraph)

  js_relevant_titles=relevant_titles.to_json()
  #js_answer=json.dumps(answer)

  print(js_relevant_titles)

if __name__=="__main__":
  main()
  






# In[28]:





# In[ ]:





# In[ ]:





# In[ ]:




