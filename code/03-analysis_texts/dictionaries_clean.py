import numpy as np
import pandas as pd
from io import StringIO
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

LM_dic = pd.read_csv("code/03-analysis_texts/Loughran-McDonald_MasterDictionary_1993-2021 (cleaned).csv", na_filter=False)
NBF_dic = pd.read_csv("code/03-analysis_texts/NBF Dictionaries (cleaned).csv", na_filter=False)

#%% organize NBF dic

NBF_val = NBF_dic.iloc[:, 5:]
NBF_val = np.repeat(np.array(NBF_val), 5, axis=0)
NBF_index = NBF_dic.iloc[:, :5]
NBF_index = pd.Series(NBF_index.values.reshape((NBF_index.values.size)))

NBF_dic_reshaped = pd.concat([pd.DataFrame(NBF_index), pd.DataFrame(NBF_val)], axis=1)
NBF_dic_reshaped.columns = ['Word', 'Sociability dictionary', 'Sociability direction', 'Ability dictionary', 'Ability direction']
NBF_dic_reshaped = NBF_dic_reshaped.drop_duplicates()
NBF_dic_reshaped["Word"] = NBF_dic_reshaped["Word"].str.lower()
NBF_dic_reshaped

# lemmatize NBF word
NBF_word = NBF_dic_reshaped['Word']
lemmatizer = WordNetLemmatizer()
NBF_lemmatized = []
for w in NBF_word:
    NBF_lemmatized.append(lemmatizer.lemmatize(w))
NBF_dic_reshaped['lemmatized'] = NBF_lemmatized
NBF_dic_reshaped = NBF_dic_reshaped.reset_index(drop=True)

# change dtype and shorten NBF dic
NBF_dic_reshaped.loc[NBF_dic_reshaped['Sociability dictionary'] == 0, 'Sociability direction'] = 0
NBF_dic_reshaped['Sociability direction'] = NBF_dic_reshaped['Sociability direction'].astype(int)
NBF_dic_reshaped.loc[NBF_dic_reshaped['Ability dictionary'] == 0, 'Ability direction'] = 0
NBF_dic_reshaped['Ability direction'] = NBF_dic_reshaped['Ability direction'].astype(int)
NBF_dic_lemma = NBF_dic_reshaped.iloc[:, [5,2,4]]
NBF_dic_lemma = NBF_dic_lemma.drop_duplicates()
NBF_dic_lemma = NBF_dic_lemma.reset_index(drop=True)

#%% organize LM dic

LM_dic['Word'] = LM_dic['Word'].str.lower()
LM_dic_cleaned = (LM_dic.iloc[:, [0,4,5,6]]).copy()

# lemmatize NBF word
LM_word = LM_dic_cleaned['Word']
lemmatizer = WordNetLemmatizer()
LM_lemmatized = []
for w in LM_word:
    LM_lemmatized.append(lemmatizer.lemmatize(w))
LM_dic_cleaned['lemmatized'] = LM_lemmatized

# shorten NBF dic
LM_dic_lemma = LM_dic_cleaned.iloc[:, [4,1,2,3]]
LM_dic_lemma = LM_dic_lemma.drop_duplicates()
LM_dic_lemma = LM_dic_lemma.reset_index(drop=True)

#%% combine two dic

dic_comb = pd.merge(left=NBF_dic_lemma, right=LM_dic_lemma, how='outer', on='lemmatized', sort=True)
dic_comb = dic_comb.fillna(0)
dic_comb.columns = ['word', 'sociability', 'ability', 'negative', 'positive', 'uncertain']
dic_comb.to_csv('code/03-analysis_texts/CombDictionary.csv')
