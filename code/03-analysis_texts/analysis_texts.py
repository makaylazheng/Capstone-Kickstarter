import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def cleanText(text):
    # lower case
    text = text.lower()

    # split sentences into words
    tokenized_word = word_tokenize(text)

    # remove puctuations
    tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]+")
    tokens = tokenizer.tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for w in tokens:
        if w not in stop_words:
            filtered_tokens.append(w)

    # stemming
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for w in filtered_tokens:
        stemmed_tokens.append(stemmer.stem(w))
    
    # lemmetization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for w in filtered_tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(w))

    return filtered_tokens, stemmed_tokens, lemmatized_tokens



def textAnalysis(texts, dic_path):
    """
    Get verbal positive, negative, warmth, ability
    input texts, output four features
    """
    # example
    # dic_path = 'code/03-analysis_texts/CombDictionary.csv'
    # LM_NBF_DictDF = pd.read_csv(dic_path, index_col=0)
    # texts_path = 'kickstarter(total)/1cover&2advertising_page.xlsx'
    # texts = pd.read_excel(texts_path, index_col=0).loc[:, "宣传活动页文字"]
    # texts = texts[0]

    # Load the dictionary
    LM_NBF_DictDF = pd.read_csv(dic_path, index_col=0)
      
    # clean texts into tokens
    tokens, stemmed_tokens, lemmatized_tokens = cleanText(texts)

    # text features
    positive_score = 0
    negative_score = 0
    uncertain_score = 0
    warmth_score = 0
    ability_score = 0
    for i in range(len(tokens)):
        count = 0
        i_pos = i_neg = i_unc = i_warm = i_abi = 0
        if tokens[i] in list(LM_NBF_DictDF['word']):
            count += 1
            scores = LM_NBF_DictDF.loc[LM_NBF_DictDF['word']==tokens[i]].mean(axis=0, numeric_only=True)
            i_pos += scores['positive']
            i_neg += scores['negative']
            i_unc += scores['uncertain']
            i_warm += scores['sociability']
            i_abi += scores['ability']
        if stemmed_tokens[i] in list(LM_NBF_DictDF['word']):
            count += 1
            scores = LM_NBF_DictDF.loc[LM_NBF_DictDF['word']==stemmed_tokens[i]].mean(axis=0, numeric_only=True)
            i_pos += scores['positive']
            i_neg += scores['negative']
            i_unc += scores['uncertain']
            i_warm += scores['sociability']
            i_abi += scores['ability']
        if lemmatized_tokens[i] in list(LM_NBF_DictDF['word']):
            count += 1
            scores = LM_NBF_DictDF.loc[LM_NBF_DictDF['word']==lemmatized_tokens[i]].mean(axis=0, numeric_only=True)
            i_pos += scores['positive']
            i_neg += scores['negative']
            i_unc += scores['uncertain']
            i_warm += scores['sociability']
            i_abi += scores['ability']
        # if any word form is found in the dictionary, summarize the scores
        if count != 0:
            positive_score += i_pos / count
            negative_score += i_neg / count
            uncertain_score += i_unc / count
            warmth_score += i_warm / count
            ability_score += i_abi / count

    return positive_score, negative_score, uncertain_score, warmth_score, ability_score, len(tokens), len(set(tokens))

#%% textual resources combination

# read texts files
cover_texts = pd.read_excel('kickstarter(total)/1cover&2advertising_page.xlsx').loc[:, ['<realpath>', '宣传活动页文字']]
faq_texts = pd.read_excel('kickstarter(total)/4FAQs.xlsx').loc[:, ['<realpath>', '答案']]
update_texts = pd.read_excel('kickstarter(total)/5update_page.xlsx').loc[:, ['<realpath>', '正文']]
# comment_texts = pd.read_excel('kickstarter(total)/6comment_page.xlsx').loc[:, ['<realpath>', '内容']]
# founder_texts = pd.read_excel('kickstarter(total)/7founder_page.xlsx').loc[:, ['<realpath>', '简介']]
# founder_mess = pd.read_excel('kickstarter(total)/8founder_message.xlsx').loc[:, ['<realpath>', '内容']]
path_csv = pd.read_csv('code/video_audio_paths.csv', index_col=0)
audio_paths = np.load('code/03-analysis_texts/speech-to-text/audio_paths.npy')
audio_texts = np.load('code/03-analysis_texts/speech-to-text/audio_texts.npy')
audio_texts = np.column_stack((audio_paths, audio_texts))
audio_texts = pd.DataFrame(audio_texts, columns=['audio_path', 'texts'])
for i in range(audio_texts.shape[0]):
    audio_texts.loc[i, 'realpath'] = path_csv[path_csv['audio_path']==audio_texts.loc[i, 'audio_path']]['realpath'].values[0]

# change column names and fill NA
cover_texts.columns = faq_texts.columns = \
    update_texts.columns = ['realpath', 'texts']
    # comment_texts.columns = \
    # founder_texts.columns = founder_mess.columns = \
cover_texts = cover_texts.fillna(' ')
update_texts = update_texts.fillna(' ')
# comment_texts = comment_texts.fillna(' ')
# founder_texts = founder_texts.fillna(' ')

# combine in a single file
cover_texts_grouped = cover_texts.groupby('realpath').apply(lambda x: ' '.join(x['texts']))
faq_texts_grouped = faq_texts.groupby('realpath').apply(lambda x: ' '.join(x['texts']))
update_texts_grouped = update_texts.groupby('realpath').apply(lambda x: ' '.join(x['texts']))
# comment_texts_grouped = comment_texts.groupby('realpath').apply(lambda x: ' '.join(x['texts']))
# founder_texts_grouped = founder_texts.groupby('realpath').apply(lambda x: ' '.join(x['texts']))
# founder_mess_grouped = founder_mess.groupby('realpath').apply(lambda x: ' '.join(x['texts']))
audio_texts_grouped = audio_texts.groupby('realpath').apply(lambda x: ' '.join(x['texts']))

# combine all texts
texts_comb = pd.concat(
    [cover_texts_grouped, faq_texts_grouped, update_texts_grouped, audio_texts_grouped]#, comment_texts_grouped, founder_texts_grouped, founder_mess_grouped]
    , axis=0
    , ignore_index=False)
texts_comb_grouped = texts_comb.groupby('realpath').apply(lambda x: ' '.join(x))

#%% analysis for texts

analysis_texts_results = pd.DataFrame()
for i in range(len(texts_comb_grouped)):
    print(i)
    text = texts_comb_grouped[i]
    positive_score, negative_score, uncertain_score, warmth_score, ability_score, len_tokens, len_tokens_unique = \
        textAnalysis(text, 'code/03-analysis_texts/CombDictionary.csv')
    analysis_texts_results.loc[i, 'realpath'] = texts_comb_grouped.index[i]
    analysis_texts_results.loc[i, 'positive'] = positive_score
    analysis_texts_results.loc[i, 'negative'] = negative_score
    analysis_texts_results.loc[i, 'uncertain'] = uncertain_score
    analysis_texts_results.loc[i, 'warmth'] = warmth_score
    analysis_texts_results.loc[i, 'ability'] = ability_score
    analysis_texts_results.loc[i, 'len_tokens'] = len_tokens
    analysis_texts_results.loc[i, 'len_tokens_uni'] = len_tokens_unique

analysis_texts_results.to_csv('code/03-analysis_texts/analysis_texts_results.csv')

