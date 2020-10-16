{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "aNl_dRhlcWDK"
   },
   "source": [
    "## Efficient Q&A and search system for Parliamentary Questions\n",
    "\n",
    "### Problem Statement: \n",
    "\n",
    " *During parliament session, each Department receives number of Parliament Questions on varied topics raised by MPs and handled in a very time bound manner on top priority. Each reply is generally prepared by seeking inputs from all the other relevant departments which requires lot of efforts and is also time consuming. It is desired a platform can be designed which can provide responses to similar PQ asked earlier, suggest probable reply and indicate different departments having similar programs and information. This will be helpful in preparing proper reply to PQ. As of now there are some search tools, separate for Lok Sabha and Rajya Sabha. But a unified, fast and effective mechanism is missing.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 523
    },
    "id": "pT1IJJvecWDM",
    "outputId": "7b49abbd-a769-426c-d296-86738c0401c6"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers in /usr/local/lib/python3.6/dist-packages (3.3.1)\n",
      "Requirement already satisfied: sacremoses in /usr/local/lib/python3.6/dist-packages (from transformers) (0.0.43)\n",
      "Requirement already satisfied: dataclasses; python_version < \"3.7\" in /usr/local/lib/python3.6/dist-packages (from transformers) (0.7)\n",
      "Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers) (3.0.12)\n",
      "Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers) (20.4)\n",
      "Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers) (2.23.0)\n",
      "Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from transformers) (1.18.5)\n",
      "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.6/dist-packages (from transformers) (4.41.1)\n",
      "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers) (2019.12.20)\n",
      "Requirement already satisfied: tokenizers==0.8.1.rc2 in /usr/local/lib/python3.6/dist-packages (from transformers) (0.8.1rc2)\n",
      "Requirement already satisfied: sentencepiece!=0.1.92 in /usr/local/lib/python3.6/dist-packages (from transformers) (0.1.91)\n",
      "Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (0.16.0)\n",
      "Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.15.0)\n",
      "Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (7.1.2)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers) (2.4.7)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (3.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2020.6.20)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (1.24.3)\n",
      "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2.10)\n",
      "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package brown to /root/nltk_data...\n",
      "[nltk_data]   Package brown is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n",
      "Requirement already satisfied: cleantext in /usr/local/lib/python3.6/dist-packages (1.1.3)\n",
      "Requirement already satisfied: nltk in /usr/local/lib/python3.6/dist-packages (from cleantext) (3.2.5)\n",
      "Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from nltk->cleantext) (1.15.0)\n"
     ]
    }
   ],
   "source": [
    "#Importing libraries\n",
    "import sys\n",
    "import spacy\n",
    "import pandas as pd\n",
    "import math\n",
    "import re\n",
    "import numpy as np\n",
    "from collections import Counter\n",
    "import gzip\n",
    "import gensim \n",
    "import logging\n",
    "import torch\n",
    "!pip install transformers\n",
    "from transformers import BertForQuestionAnswering\n",
    "from transformers import BertTokenizer\n",
    "import json\n",
    "\n",
    "import nltk\n",
    "from nltk.corpus import wordnet as wn\n",
    "from nltk.corpus import brown\n",
    "nltk.download('punkt')\n",
    "nltk.download('brown')\n",
    "nltk.download('wordnet')\n",
    "\n",
    "from itertools import product\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "!pip install cleantext\n",
    "from cleantext import clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "ngtROPPWjicL"
   },
   "outputs": [],
   "source": [
    "#Fetching the data\n",
    "data=pd.read_csv(\"sample_data/rajyasabha_questions_and_answers_2009.csv\")\n",
    "questions=data['question_description']\n",
    "answers=data['answer']\n",
    "titles=data['question_title']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RBVfAuepcWDm"
   },
   "source": [
    "## Step 1: Creating a cosine similarity calculator\n",
    "\n",
    "This helps in calculating similarity between two sentences. Since we are planning to take a) Title b) Question as input from the user, using the 'Title' we will first try to get the relevant rows to search through.\n",
    "\n",
    "![title](img/step1.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "7NZmc91wjicR"
   },
   "outputs": [],
   "source": [
    "#WORD NET MODULES\n",
    "ALPHA = 0.2\n",
    "BETA = 0.45\n",
    "ETA = 0.4\n",
    "PHI = 0.2\n",
    "DELTA = 0.85\n",
    "\n",
    "def get_best_synset_pair(word_1, word_2):\n",
    "    \n",
    "    max_sim = -1.0\n",
    "    synsets_1 = wn.synsets(word_1)\n",
    "    synsets_2 = wn.synsets(word_2)\n",
    "    if len(synsets_1) == 0 or len(synsets_2) == 0:\n",
    "        return None, None\n",
    "    else:\n",
    "        max_sim = -1.0\n",
    "        best_pair = None, None\n",
    "        for synset_1 in synsets_1:\n",
    "            for synset_2 in synsets_2:\n",
    "               sim = wn.path_similarity(synset_1, synset_2)\n",
    "               if sim!=None and sim > max_sim:\n",
    "                   max_sim = sim\n",
    "                   best_pair = synset_1, synset_2\n",
    "        return best_pair\n",
    "    \n",
    "def length_dist(synset_1, synset_2):\n",
    "    \n",
    "    l_dist = sys.maxsize\n",
    "    if synset_1 is None or synset_2 is None: \n",
    "        return 0.0\n",
    "    if synset_1 == synset_2:\n",
    "        # if synset_1 and synset_2 are the same synset return 0\n",
    "        l_dist = 0.0\n",
    "    else:\n",
    "        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        \n",
    "        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])\n",
    "        if len(wset_1.intersection(wset_2)) > 0:\n",
    "            # if synset_1 != synset_2 but there is word overlap, return 1.0\n",
    "            l_dist = 1.0\n",
    "        else:\n",
    "            # just compute the shortest path between the two\n",
    "            l_dist = synset_1.shortest_path_distance(synset_2)\n",
    "            if l_dist is None:\n",
    "                l_dist = 0.0\n",
    "    # normalize path length to the range [0,1]\n",
    "    return math.exp(-ALPHA * l_dist)\n",
    "\n",
    "\n",
    "def hierarchy_dist(synset_1, synset_2):\n",
    "    \n",
    "    h_dist = sys.maxsize\n",
    "    if synset_1 is None or synset_2 is None: \n",
    "        return h_dist\n",
    "    if synset_1 == synset_2:\n",
    "        # return the depth of one of synset_1 or synset_2\n",
    "        h_dist = max([x[1] for x in synset_1.hypernym_distances()])\n",
    "    else:\n",
    "        # find the max depth of least common subsumer\n",
    "        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}\n",
    "        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}\n",
    "        lcs_candidates = set(hypernyms_1.keys()).intersection(\n",
    "            set(hypernyms_2.keys()))\n",
    "        if len(lcs_candidates) > 0:\n",
    "            lcs_dists = []\n",
    "            for lcs_candidate in lcs_candidates:\n",
    "                lcs_d1 = 0\n",
    "                if lcs_candidate in hypernyms_1:\n",
    "                    lcs_d1 = hypernyms_1[lcs_candidate]\n",
    "                lcs_d2 = 0\n",
    "                if lcs_candidate in hypernyms_2:\n",
    "                    lcs_d2 = hypernyms_2[lcs_candidate]\n",
    "                lcs_dists.append(max([lcs_d1, lcs_d2]))\n",
    "            h_dist = max(lcs_dists)\n",
    "        else:\n",
    "            h_dist = 0\n",
    "    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / \n",
    "        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))\n",
    "\n",
    "def word_similarity(word_1, word_2):\n",
    "    synset_pair = get_best_synset_pair(word_1, word_2)\n",
    "    return (length_dist(synset_pair[0], synset_pair[1]) * \n",
    "        hierarchy_dist(synset_pair[0], synset_pair[1]))\n",
    "\n",
    "def most_similar_word(word, word_set):\n",
    "    max_sim = -1.0\n",
    "    sim_word = \"\"\n",
    "    for ref_word in word_set:\n",
    "      sim = word_similarity(word, ref_word)\n",
    "      if sim > max_sim:\n",
    "          max_sim = sim\n",
    "          sim_word = ref_word\n",
    "    return sim_word, max_sim\n",
    "\n",
    "brown_freqs = dict()\n",
    "N=0\n",
    "def info_content(lookup_word):\n",
    "    \n",
    "    global N\n",
    "    if N == 0:\n",
    "        # poor man's lazy evaluation\n",
    "        for sent in brown.sents():\n",
    "            for word in sent:\n",
    "                word = word.lower()\n",
    "                if word not in brown_freqs:\n",
    "                    brown_freqs[word] = 0\n",
    "                brown_freqs[word] = brown_freqs[word] + 1\n",
    "                N = N + 1\n",
    "    lookup_word = lookup_word.lower()\n",
    "    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]\n",
    "    return 1.0 - (math.log(n + 1) / math.log(N + 1))\n",
    "\n",
    "def semantic_vector(words, joint_words, info_content_norm):\n",
    "    sent_set = set(words)\n",
    "    semvec = np.zeros(len(joint_words))\n",
    "    i = 0\n",
    "    for joint_word in joint_words:\n",
    "        if joint_word in sent_set:\n",
    "            # if word in union exists in the sentence, s(i) = 1 (unnormalized)\n",
    "            semvec[i] = 1.0\n",
    "            if info_content_norm:\n",
    "                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)\n",
    "        else:\n",
    "            # find the most similar word in the joint set and set the sim value\n",
    "            sim_word, max_sim = most_similar_word(joint_word, sent_set)\n",
    "            semvec[i] = PHI if max_sim > PHI else 0.0\n",
    "            if info_content_norm:\n",
    "                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)\n",
    "        i = i + 1\n",
    "    return semvec       \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "UGjCAPZbcWDn"
   },
   "outputs": [],
   "source": [
    "#Function to find similarity between two strings that are converted into vectors\n",
    "WORD = re.compile(r\"\\w+\")\n",
    "\n",
    "def get_cosine(sentence_1, sentence_2):\n",
    "    sentence_1 = re.sub('[^A-Za-z0-9\\s]', '', sentence_1).lower()\n",
    "    sentence_2 = re.sub('[^A-Za-z0-9\\s]', '', sentence_2).lower()\n",
    "    info_content_norm = True\n",
    "    words_1 = nltk.word_tokenize(sentence_1)\n",
    "    words_2 = nltk.word_tokenize(sentence_2)\n",
    "    joint_words = set(words_1).union(set(words_2))\n",
    "    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)\n",
    "    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)\n",
    "    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "rN0zU2NUcWDy"
   },
   "outputs": [],
   "source": [
    "#Function to get top 100 relevant rows to search\n",
    "def get_relevant_indices(query,title_list):\n",
    "    \n",
    "    cosine_list=np.empty(len(title_list))\n",
    "    for i in range(len(title_list)):\n",
    "        cosine_val_i=get_cosine(query,title_list[i])\n",
    "        cosine_list[i]=cosine_val_i\n",
    "    \n",
    "    relevant_indices=cosine_list.argsort()[-10:][::-1]\n",
    "    return relevant_indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "B7LtoigCcWD2"
   },
   "outputs": [],
   "source": [
    "#Function to get relevant titles\n",
    "def get_relevant_titles(query):\n",
    "    titles=data['question_title'].values.tolist()\n",
    "    relevant_indices=get_relevant_indices(query,titles)\n",
    "    relevant_titles=[]\n",
    "    for i in range(len(relevant_indices)):\n",
    "        relevant_titles.append(titles[relevant_indices[i]])\n",
    "    \n",
    "    relevant_titles_df=pd.DataFrame({'Indices':relevant_indices,\n",
    "                                    'Titles':relevant_titles})\n",
    "    return relevant_titles_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 503
    },
    "id": "5RDuMSVXcWD6",
    "outputId": "5784b1ed-f028-4402-94d3-9cab35034d29"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Indices                                        Titles\n",
      "0       91                      SELLING STAKES OF PSES .\n",
      "1      168                  FALL IN REVENUE COLLECTION .\n",
      "2      217                         RESTATEMENT OF LAWS .\n",
      "3      224           STATUS OF LAW COMMISSIONS REPORTS .\n",
      "4      208                    PROPERTY RIGHTS OF WIVES .\n",
      "5      170  INCREASE IN STAMP DUTY ON INSURANCE PRODUCTS\n",
      "6      187                       TAX EVASION BY BUILDERS\n",
      "7      157                      CORE BANKING SOLUTIONS .\n",
      "8      159        PENALTY FOR REPAYING LOAN IN ADVANCE .\n",
      "9      112                           SALE PRICE OF GAS .\n",
      "\n",
      "\n",
      " For better visualization\n",
      "\n",
      "\n",
      "Relevant Indices: \n",
      "\n",
      "[ 91 168 217 224 208 170 187 157 159 112]\n",
      "\n",
      "Relevant Titles:\n",
      "\n",
      "['SELLING STAKES OF PSES .' 'FALL IN REVENUE COLLECTION .'\n",
      " 'RESTATEMENT OF LAWS .' 'STATUS OF LAW COMMISSIONS REPORTS .'\n",
      " 'PROPERTY RIGHTS OF WIVES .'\n",
      " 'INCREASE IN STAMP DUTY ON INSURANCE PRODUCTS' 'TAX EVASION BY BUILDERS'\n",
      " 'CORE BANKING SOLUTIONS .' 'PENALTY FOR REPAYING LOAN IN ADVANCE .'\n",
      " 'SALE PRICE OF GAS .']\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 228
    },
    "id": "uiBvwAQTjico",
    "outputId": "d4fc1885-5444-470c-d6d1-834bf8e9a573"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "THE MINISTER OF STATE IN THE MINISTRY OF FINANCE (S.S. PALANIMANICKAM) (a)&(b)The policy on disinvestment articulated in the President`s Speech to Joint Session of Parliament on 4th June, 2009 and Finance Minister`s Budget Speech on 6th July, 2009 requires the development of `people ownership` of Central Public Sector Undertakings (CPSUs) to share in their wealth and prosperity, with Government retaining majority shareholding and control. This objective is relevant to profit-earning CPSUs as it is only these that will sustain investor-interest for sharing in their prosperity. In line with this policy announcement, Government has decided that: (i) already listed profitable CPSUs not meeting the mandatory public shareholding of 10% are to be made compliant; and (ii) all CPSUs having positive networth, no accumulated losses and having earned net profit for three preceding consecutive years, are to be listed through public offerings out of Government shareholding or issue of fresh equity by the company or a combination of both. (c) No target for revenue generation from disinvestment has been fixed. (d) The proceeds from disinvestment would be channelised into National Investment Fund and during April 2009 to March 2012 would be available in full for meeting the capital expenditure requirements of selected Social Sector Programmes decided by the Planning Commission/Department of Expenditure.\n",
      "MINISTER OF STATE IN THE MINISTRY OF FINANCE (SHRI S.S. PALANIMANICKAM) (a) & (b): During the Financial Year 2008-09, Government`s revenue collection was as follows: (Rs.in crore) S.No. Nature of Duty/ Budget Revised Actual collection Tax Estimate Estimate (Provisional) A Direct Taxes 3,65,000 3,45,000 3,38,212 B Indirect Taxes 3,21,264 2,81,359 2,69879 C Total 6,86,264 6,26,359 6,08,091 With actual collection of Rs.6,08,091 crore against the Revised Estimate of Rs.6,26,359 crore, the revenue shortfall was Rs. 18,268 crore. The shortfall in collection of Indirect Taxes was on account of the Government`s foregoing of revenue of over Rs.40,000 crore to provide fiscal stimulus to the economy, as also the economic slowdown resulting from the global financial meltdown and consequent economic recession in developed economies. The economic slowdown dented the profits of companies and also resulted in lesser salary payouts, resulting in decrease in collection in Corporate Tax and Personal Income Tax. The stock market also remained subdued which impacted the collection of Securities Transaction Tax. Therefore, direct tax collections were below the Revised Estimate. (c): So far as indirect taxes, are concerned, no specific industry wise economic relief has been provided. With regard to direct taxes, any undertaking engaged in carrying on the `specified business�<U+0080>� referred to in the various tax-relief provisions of the Income-tax Act is eligible for the tax benefit thereunder subject to the conditions specified therein Therefore it is not possible to provide the names of any public or private company likely to benefit from such provisions of the Income Tax Act.\n",
      "MINISTER OF LAW AND JUSTICE (DR. M. VEERAPPA MOILY) (a): Government has not received any such information. (b): Yes, Sir. The Hon�<U+0080><U+0099>ble Chief Justice of India who is an Ex-Officio President of the Indian Law Institute has constituted a Restatement of Law Project Committee consisting of Judges, senior advocates and academicians to undertake a research project on Restatement of Law on various topics. (c): The Restatement of Law Project Committee has initially selected following three subjects as a pilot project in order to create models for future use: (i) The Legislation Privileges (ii) Contempt of Court (iii) Public Interest Litigation\n",
      "MINISTER OF LAW AND JUSTICE (DR. M. VEERAPPA MOILY) (a) No, Sir. (b) Does not arise. (c) The Reports of the Law Commission are still laid on the Table of both the Houses of Parliament (d) & (e) Does not arise.\n",
      "MINISTER OF LAW AND JUSTICE (DR. M. VEERAPPA MOILY) (a) to (c) The information is being collected and will be laid on the table of the House.\n",
      "MINISTER OF STATE IN THE MINISTRY OF FINANCE (SHRI S.S. PALANIMANICKAM) (a) : No, Sir. (b) to (d): Does not arise.\n",
      "MINISTER OF STATE IN THE MINISTRY OF FINANCE (SHRI S.S.PALANIMANICKAM) (a): Yes Sir. Certain cases of tax evasion by builders have come to the notice of Income-tax Department. (b) to (d):;The Income-tax Department has conducted Search and Seizure operations under Section 132 of the Income tax Act, 1961 in total 182 cases of builders and developers all over the country during the last three Financial Years. The operations have led to detection of total undisclosed income of Rs 3541.38 Crore.\n",
      " (b) the name of these banks and the details of reasons as due to which these banks are not able to provide CBS\n",
      "THE MINISTER OF STATE IN THE MINISTRY OF FINANCE (SHRI NAMO NARAIN MEENA) (a) to (d):- Reserve Bank of India (RBI) has reported that pre-pay raent/foreclosure charges are normally levied by banks as pre-payment of loans affects their Asset Liability Management. RBI has not issued any` guidelines `regarding pre-payment/foreclosure charges of loans. However, some of the banks have reported that they do not levy any penalty for repaying the loan in advance, if the loan is repaid by the borrower out of his/her own sources/internal: accruals. (e):- In terms of extant instructions, in the context of granting greater functional autonomy to banks, operational freedom has been given to scheduled commercial banks on all matters pertaining to banking transactions, including pre-payment/ foreclosure charges on loans. With effect from September 7, 1999, banks have been given freedom to fix service charges for various types of services rendered by them. While fixing service charges, banks should ensure that ther charges are reasonable and not but of line with the average cost of providing these services. In order to ensure transparency, banks have been advised to display and update on their websites the details of various service charges in a prescribed format. Further in terms of the `Guidelines of Fair Practices Code for Lenders` banks have been advised by RBI that loan application forms should be comprehensive and should include information about the fees/charges, if any; payable for processing, the amount of such fees refundable in the case of non-acceptance of applications, pre-payment options and any other matter which affects the interest of the borrower, sq that a meaningful comparison with other banks can be made and an informed decision can be taken by the borrower.\n",
      "MINISTER OF STATE IN THE MINISTRY OF FINANCE (SHRI S.S. PALANIMANICKAM) (a) to (b): The information is being collected and will be laid on the Table of the House.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#paragraph to search for the answer\n",
    "def get_paragraph(relevant_titles):\n",
    "  paragraph=''\n",
    "  for i in relevant_titles['Indices'].values:\n",
    "    paragraph+=answers[i]+'\\n'\n",
    "  return paragraph\n",
    "\n",
    "paragraph=get_paragraph(relevant_titles)\n",
    "\n",
    "print(paragraph)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 272,
     "referenced_widgets": [
      "e7a0b0905828445eb52f33613b2242e5",
      "2b7f0017d0f3412eb3e0dca98e2aa043",
      "4ae85d782eef49429af4a1fc859f908a",
      "b91ba5578c5c45e9808114cc8c67bfbc",
      "ab6d78e95a9749f1bc8879d0dc2702a8",
      "c4e4d873becc481a9170157e3bd6389e",
      "117ac84ba5f447d9b8836ae7cd1ee51b",
      "472366c5137b40068e02d6a3fd60e665",
      "363eb92233d549f3ade729b758db8f44",
      "23e561edb7d04cdbbe62a85aa201b4a3",
      "ebbc95440a194b1881ca794b25bfd459",
      "71085fc4cfb4405faa127866d82064d5",
      "4b0ce2ba4240437784eb4e8325000928",
      "a7d8f6ca0e2344bf92b9df6e45433102",
      "1657a413ac374d1793af39e6828876ff",
      "1060fefac2a74a4399aff6571f75fcb1",
      "f4aa4fff7793492586f2d386c23535b5",
      "6328ae48a9214ffc959dba482bc7b30b",
      "3eb135aec1994b63bfe6c775e4d517a6",
      "b85663a51d4e4f01ad8c0961dbe70a09",
      "bad7d5add20241dd8bdc1a25be8814ba",
      "7226f8be55d7427bb6e815c866c226ec",
      "d5c803ca7989403d9bcafda80ded950e",
      "d6bb6a8868604a948c23302b31138d43"
     ]
    },
    "id": "F6fKIfTucWD_",
    "outputId": "606feba2-aceb-4137-bb30-7787570503e2",
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e7a0b0905828445eb52f33613b2242e5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(FloatProgress(value=0.0, description='Downloading', max=433.0, style=ProgressStyle(description_…"
      ]
     },
     "metadata": {
      "tags": []
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "363eb92233d549f3ade729b758db8f44",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(FloatProgress(value=0.0, description='Downloading', max=440473133.0, style=ProgressStyle(descri…"
      ]
     },
     "metadata": {
      "tags": []
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForQuestionAnswering: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']\n",
      "- This IS expected if you are initializing BertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n",
      "- This IS NOT expected if you are initializing BertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).\n",
      "Some weights of BertForQuestionAnswering were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f4aa4fff7793492586f2d386c23535b5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(FloatProgress(value=0.0, description='Downloading', max=231508.0, style=ProgressStyle(descripti…"
      ]
     },
     "metadata": {
      "tags": []
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "#Importing the BERT pretrained on SQuAD\n",
    "\n",
    "#Model\n",
    "model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')\n",
    "\n",
    "#Tokenizer\n",
    "tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "NZV0g7mIcWEF"
   },
   "outputs": [],
   "source": [
    "#Answer prediction function\n",
    "def get_answer(query,paragraph):\n",
    "    \n",
    "    #Don't worry much about these lines. These will just convert the input\n",
    "    #into a form that the BERT model can accept\n",
    "    \n",
    "    encoding = tokenizer.encode_plus(text=query,text_pair=paragraph,max_length=512,truncation=True)\n",
    "    inputs = encoding['input_ids']  #Token embeddings\n",
    "    sentence_embedding = encoding['token_type_ids']  #Segment embeddings\n",
    "    tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens\n",
    "    \n",
    "    #Feeding the input into the model\n",
    "    start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))\n",
    "    \n",
    "    #Bert assigns scores to each token. The tokens that have the highest\n",
    "    #Start and end scores are likely to be our starting and ending of the\n",
    "    #answer. Below lines of code will do that. For better understanding, refer\n",
    "    #the functioning of BERT online\n",
    "    \n",
    "    start_index = torch.argmax(start_scores)\n",
    "    end_index = torch.argmax(end_scores)\n",
    "    answer = ' '.join(tokens[start_index:end_index+1])\n",
    "    \n",
    "    #BERT uses word peice tokenization. i.e. playing = play##ing. To make our\n",
    "    #output look neat, let's removes those ##\n",
    "    \n",
    "    corrected_answer = ''\n",
    "    for word in answer.split():\n",
    "        #If it's a subword token\n",
    "        if word[0:2] == '##':\n",
    "            corrected_answer += word[2:]\n",
    "        else:\n",
    "            corrected_answer += ' ' + word\n",
    "    \n",
    "    return corrected_answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 55
    },
    "id": "fc3RP4WwcWEJ",
    "outputId": "46b6eb7b-708d-4626-a04e-f3a525640bc8"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " speech on 6th july , 2009 requires the development of ` people ownership ` of central public sector undertakings ( cpsus ) to share in their wealth and prosperity , with government retaining majority shareholding and control . this objective is relevant to profit - earning cpsus as it is only these that will sustain investor - interest for sharing in their prosperity . in line with this policy announcement , government has decided that : ( i ) already listed profitable cpsus not meeting the mandatory public shareholding of 10 % are to be made compliant ; and ( ii ) all cpsus having positive networth , no accumulated losses and having earned net profit for three preceding consecutive years , are to be listed through public offerings out of government shareholding or issue of fresh equity by the company or a combination of both . ( c ) no target for revenue generation from disinvestment has been fixed . ( d ) the proceeds from disinvestment would be channelised into national investment fund and during april 2009 to march 2012 would be available in full for meeting the capital expenditure requirements of selected social sector programmes decided by the planning commission / department of expenditure . minister of state in the ministry of finance ( shri s . s . palanimanickam ) ( a ) & ( b ) : during the financial year 2008 - 09 , government ` s revenue collection was as follows : ( rs . in crore ) s . no . nature of duty / budget revised actual collection tax estimate estimate ( provisional ) a direct taxes 3 , 65 , 000 3 , 45 , 000 3 , 38 , 212 b indirect taxes 3 , 21 , 264 2 , 81 , 359 2 , 69879 c total 6 , 86 , 264 6 , 26 , 359 6 , 08 , 091 with actual\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 52
    },
    "id": "ivTUVjdQlmN9",
    "outputId": "11b0b24f-7efa-4df1-e117-d46c6eff913e"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'get_paragraph' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-10-d4c745861556>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m\u001b[0;34m==\u001b[0m\u001b[0;34m\"__main__\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m   \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-10-d4c745861556>\u001b[0m in \u001b[0;36mmain\u001b[0;34m()\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m   \u001b[0;31m#creating a merged paragraph to be searched\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 9\u001b[0;31m   \u001b[0mparagraph\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mget_paragraph\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrelevant_titles\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     10\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m   \u001b[0;31m#searching the paragraph for a suitable answer using bert\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'get_paragraph' is not defined"
     ]
    }
   ],
   "source": [
    "#main function\n",
    "def main():\n",
    "    \n",
    "  title=sys.argv[1]\n",
    "  #fetching relevant titles\n",
    "  relevant_titles=get_relevant_titles(title)\n",
    "\n",
    "  #creating a merged paragraph to be searched\n",
    "  #paragraph=get_paragraph(relevant_titles)\n",
    "\n",
    "  #searching the paragraph for a suitable answer using bert\n",
    "  #answer=get_answer(question,paragraph)\n",
    "\n",
    "  js_relevant_titles=relevant_titles.to_json()\n",
    "  #js_answer=json.dumps(answer)\n",
    "\n",
    "  return js_relevant_titles\n",
    "\n",
    "if __name__==\"__main__\":\n",
    "  main()\n",
    "  \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 107
    },
    "id": "XT-ZbfAwlmEe",
    "outputId": "c9311e64-3d0a-4318-cfc2-96d6f0891351",
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'str'>\n",
      "<class 'str'>\n",
      "{\"Indices\":{\"0\":187,\"1\":178,\"2\":123,\"3\":182,\"4\":102,\"5\":88,\"6\":150,\"7\":270,\"8\":213,\"9\":246},\"Titles\":{\"0\":\"TAX EVASION BY BUILDERS\",\"1\":\"EVASION OF INCOME TAX BY CHEMISTS\",\"2\":\"TAX CONCESSION TO STATE .\",\"3\":\"LOSSES OF BANKS\",\"4\":\"LOAN DISBURSAL UNDER SELF EMPLOYMENT SCHEMES\",\"5\":\"SEPARATION OF LENDING BUSINESS FROM INVESTMENT BUSINESS BY BANKS .\",\"6\":\"MERGER AND ACQUISITIONS OF PUBLIC SECTOR BANKS .\",\"7\":\"INCOME TAX SLABS FOR INDUSTRIES\",\"8\":\"DISCLOSURE OF WEALTH BY JUDGES .\",\"9\":\"DISCLOSURE OF ASSETS BY JUDGES .\"}}\n",
      "\" search and seizure actions under section 132 of income tax act 1961 have been conducted in case of one chemist at chandigarh and one manufacturer of drugs and medicines at meerut . however , no search and seizure action has been conducted in case of any , chemist at faridabad . ( d ) does not arise in view of the answer to part ( c ) above . minister of state in the ministry of finance ( shri s . s . palanimanickam ) ( a ) : yes , sir . in respect of direct\"\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "wrwYS2QFhj4w"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "HtEdV-6Qjic8",
    "outputId": "1f5d7b6a-995a-484d-f8cb-9e72d506a49a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "minister of commerce and industry (shri anand sharma) a) to d): a statement is laid on the table of the house. statement referred to in reply to parts to of rajya sabha starred question no. 396 for answer on 16th december, 2009 regarding �<u+0080><u+009c>spurt in prices of gold �<u+0080>� yes, sir. increase in prices of gold in the international markets, seasonal demand by major consumers and investment buying are the major factors known to affect the prices of gold. & : the gold prices are broadly driven by the international gold prices. government has minimal control over them.\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "gAywmuM3jidB"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Notebook_Rakesh.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "1060fefac2a74a4399aff6571f75fcb1": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "117ac84ba5f447d9b8836ae7cd1ee51b": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "1657a413ac374d1793af39e6828876ff": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "23e561edb7d04cdbbe62a85aa201b4a3": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "2b7f0017d0f3412eb3e0dca98e2aa043": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "363eb92233d549f3ade729b758db8f44": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_ebbc95440a194b1881ca794b25bfd459",
       "IPY_MODEL_71085fc4cfb4405faa127866d82064d5"
      ],
      "layout": "IPY_MODEL_23e561edb7d04cdbbe62a85aa201b4a3"
     }
    },
    "3eb135aec1994b63bfe6c775e4d517a6": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "Downloading: 100%",
      "description_tooltip": null,
      "layout": "IPY_MODEL_7226f8be55d7427bb6e815c866c226ec",
      "max": 231508,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_bad7d5add20241dd8bdc1a25be8814ba",
      "value": 231508
     }
    },
    "472366c5137b40068e02d6a3fd60e665": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "4ae85d782eef49429af4a1fc859f908a": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "Downloading: 100%",
      "description_tooltip": null,
      "layout": "IPY_MODEL_c4e4d873becc481a9170157e3bd6389e",
      "max": 433,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_ab6d78e95a9749f1bc8879d0dc2702a8",
      "value": 433
     }
    },
    "4b0ce2ba4240437784eb4e8325000928": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": "initial"
     }
    },
    "6328ae48a9214ffc959dba482bc7b30b": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "71085fc4cfb4405faa127866d82064d5": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_1060fefac2a74a4399aff6571f75fcb1",
      "placeholder": "​",
      "style": "IPY_MODEL_1657a413ac374d1793af39e6828876ff",
      "value": " 440M/440M [00:13&lt;00:00, 32.6MB/s]"
     }
    },
    "7226f8be55d7427bb6e815c866c226ec": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "a7d8f6ca0e2344bf92b9df6e45433102": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "ab6d78e95a9749f1bc8879d0dc2702a8": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": "initial"
     }
    },
    "b85663a51d4e4f01ad8c0961dbe70a09": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_d6bb6a8868604a948c23302b31138d43",
      "placeholder": "​",
      "style": "IPY_MODEL_d5c803ca7989403d9bcafda80ded950e",
      "value": " 232k/232k [00:00&lt;00:00, 1.83MB/s]"
     }
    },
    "b91ba5578c5c45e9808114cc8c67bfbc": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_472366c5137b40068e02d6a3fd60e665",
      "placeholder": "​",
      "style": "IPY_MODEL_117ac84ba5f447d9b8836ae7cd1ee51b",
      "value": " 433/433 [00:00&lt;00:00, 1.53kB/s]"
     }
    },
    "bad7d5add20241dd8bdc1a25be8814ba": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": "initial"
     }
    },
    "c4e4d873becc481a9170157e3bd6389e": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "d5c803ca7989403d9bcafda80ded950e": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "d6bb6a8868604a948c23302b31138d43": {
     "model_module": "@jupyter-widgets/base",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "e7a0b0905828445eb52f33613b2242e5": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_4ae85d782eef49429af4a1fc859f908a",
       "IPY_MODEL_b91ba5578c5c45e9808114cc8c67bfbc"
      ],
      "layout": "IPY_MODEL_2b7f0017d0f3412eb3e0dca98e2aa043"
     }
    },
    "ebbc95440a194b1881ca794b25bfd459": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "Downloading: 100%",
      "description_tooltip": null,
      "layout": "IPY_MODEL_a7d8f6ca0e2344bf92b9df6e45433102",
      "max": 440473133,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_4b0ce2ba4240437784eb4e8325000928",
      "value": 440473133
     }
    },
    "f4aa4fff7793492586f2d386c23535b5": {
     "model_module": "@jupyter-widgets/controls",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_3eb135aec1994b63bfe6c775e4d517a6",
       "IPY_MODEL_b85663a51d4e4f01ad8c0961dbe70a09"
      ],
      "layout": "IPY_MODEL_6328ae48a9214ffc959dba482bc7b30b"
     }
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
