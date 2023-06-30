import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#%% independent variables

path_csv = pd.read_csv('code/video_audio_paths.csv', index_col=0)
video_index = path_csv['orgPath_trun'].notna()
audio_index = path_csv['audio_path'].notna()

text_features = pd.read_csv('code/03-analysis_texts/analysis_texts_results.csv', index_col=0)
text_features['Text-Warmth'] = text_features['warmth'] / text_features['len_tokens']
text_features['Text-Ability'] = text_features['ability'] / text_features['len_tokens']
text_features['Text-Positive'] = text_features['positive'] / text_features['len_tokens']
text_features['Text-Negative'] = text_features['negative'] / text_features['len_tokens']

audio_features = pd.read_csv('code/02-analysis_audio/analysis_audio_results.csv', index_col=0)
audio_features.loc[audio_index, :] = audio_features.loc[audio_index, :].fillna(0)

video_features = pd.read_csv('code/01-analysis_video/analysis_video_results.csv', index_col=0)
video_features.loc[video_index, :] = video_features.loc[video_index, :].fillna(0)
video_features['Video-Positive'] = video_features['happy'] / 100
video_features['Video-Negative'] = video_features[['angry', 'sad', 'fear', 'disgust']].sum(axis=1) / 100

temp = pd.merge(text_features, audio_features, on='realpath')
ind_var = pd.merge(temp, video_features, on='realpath')
ind_var

#%% dependent variables

cover_info = pd.read_excel('kickstarter(total)/1cover&2advertising_page.xlsx')
update_info = pd.read_excel('kickstarter(total)/5update_page.xlsx')

# done_amount
amount = cover_info[['<realpath>', '已筹金额', '目标金额']].copy()
amount.columns = ['realpath', 'done_amount', 'goal_amount']
def remove_nonDigit(s):
    s_prime = ''
    for i in s:
        if i.isnumeric() is True:
            s_prime += i
    return s_prime
amount['done_amount'] = amount['done_amount'].apply(remove_nonDigit)
amount['goal_amount'] = amount['goal_amount'].apply(remove_nonDigit)
amount[['done_amount', 'goal_amount']] = amount[['done_amount', 'goal_amount']].astype(int)
amount['status'] = (amount['done_amount'] >= amount['goal_amount']).astype(int)

# investing population
population = cover_info[['<realpath>', '支持者人数']].copy()
population.columns = ['realpath', 'population']
population['population'] = population['population'].apply(remove_nonDigit)
population['population'] = population['population'].astype(int)

# posterior success
pos_url = cover_info[['<realpath>', '成功项目主页URL']].copy()
pos_url.columns = ['realpath', 'url']
pd.isna(pos_url.iloc[311, 1])
pos_url['pos_suc_url'] = (pos_url['url'].apply(pd.isna)).astype(int)

temp = pd.merge(amount, population, on='realpath')
dep_var = pd.merge(temp, pos_url, on='realpath')
dep_var.columns

#%% control variables

# founder info
founder_csv = pd.read_excel('kickstarter(total)/7founder_page.xlsx')
founder_hist = founder_csv[['<realpath>', '业绩']].copy()
def remove_split(s):
    s_list = s.split()
    nums = []
    for i in s_list:
        if i == 'First':
            nums.append(0)
            continue
        try:
            num = int(i)
            nums.append(num)
        except:
            continue
    return nums
founder_hist['nums'] = founder_hist.iloc[:,1].apply(remove_split)
founder_hist['created_num'] = founder_hist['nums'].apply(lambda x: x[0])
founder_hist['backed_num'] = founder_hist['nums'].apply(lambda x: x[1])
founder_hist = founder_hist.groupby('<realpath>').mean()

founder_coll = founder_csv[['<realpath>', '协作者']].copy()
founder_coll = founder_coll.groupby('<realpath>').count()
founder_coll.columns = ['coll_num']

founder_info = pd.merge(founder_hist, founder_coll, right_index=True, left_index=True)
founder_info = founder_info.reset_index()
founder_info.columns = ['realpath'] + list(founder_info.columns[1:])

# %% summary statistics
temp = pd.merge(dep_var, ind_var, on='realpath')
df_reg = pd.merge(temp, founder_info, on='realpath', how='left')
pitch_features = ['Text-Warmth', 'Text-Ability', 'Text-Positive', 'Text-Negative'
                 , 'Vocal-Arousal', 'Vocal-Valence', 'Vocal-Positive', 'Vocal-Negative'
                 , 'Video-Positive', 'Video-Negative'
                  # control
                 , 'backed_num', 'coll_num', 'len_tokens']
notna_index = ~ df_reg[pitch_features].isnull().T.any()
df_reg = df_reg.loc[notna_index]
with pd.option_context('display.max_rows', 10,
                       'display.max_columns', None,
                       'display.width', 1000,
                       'display.precision', 2,
                       'display.colheader_justify', 'left'):
    display(df_reg[list(dep_var.columns[1:])+pitch_features].describe())
print(df_reg[list(dep_var.columns[1:])+pitch_features].describe().T.to_latex(float_format="%.2f"))

# %% correaltion heatmap
temp = pd.merge(dep_var, ind_var, on='realpath')
df_reg = pd.merge(temp, founder_info, on='realpath', how='left')
pitch_features = ['Text-Warmth', 'Text-Ability', 'Text-Positive', 'Text-Negative'
                 , 'Vocal-Arousal', 'Vocal-Valence', 'Vocal-Positive', 'Vocal-Negative'
                 , 'Video-Positive', 'Video-Negative'
                  # control
                 , 'backed_num', 'coll_num', 'len_tokens']
notna_index = ~ df_reg[pitch_features].isnull().T.any()
df_reg = df_reg.loc[notna_index]

# max min standardization
for i in pitch_features+list(dep_var.columns[1:]):
    max_i = df_reg[i].max()
    min_i = df_reg[i].min()
    df_reg[i] = df_reg[i].apply(lambda x: (x - min_i) / (max_i - min_i))

# correlation
plt.subplots(figsize=(12,12),dpi=1080,facecolor='w')
fig=sns.heatmap(df_reg[pitch_features].corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
fig

#%% VIF
temp = pd.merge(dep_var, ind_var, on='realpath')
df_reg = pd.merge(temp, founder_info, on='realpath', how='left')
pitch_features = ['Text-Warmth', 'Text-Ability','Text-Positive', 'Text-Negative'
                 , 'Vocal-Valence', 'Vocal-Positive', 'Vocal-Negative'
                 , 'Video-Positive', 'Video-Negative'
                 # control
                 , 'backed_num', 'coll_num', 'len_tokens']
notna_index = ~ df_reg[pitch_features].isnull().T.any()
df_reg = df_reg.loc[notna_index]

# max min standardization
for i in pitch_features:
    max_i = df_reg[i].max()
    min_i = df_reg[i].min()
    df_reg[i] = df_reg[i].apply(lambda x: (x - min_i) / (max_i - min_i))

# multicollinearity
vif = [variance_inflation_factor(df_reg[pitch_features].values, df_reg[pitch_features].columns.get_loc(i)) for i in df_reg[pitch_features].columns]
vif


#%% ultimate dr_reg
temp = pd.merge(dep_var, ind_var, on='realpath')
df_reg = pd.merge(temp, founder_info, on='realpath', how='left')
pitch_features = ['Text-Ability','Text-Positive', 'Text-Negative'
                 , 'Vocal-Positive', 'Vocal-Negative'
                 , 'Video-Positive', 'Video-Negative'
                 # control
                 , 'backed_num', 'coll_num', 'len_tokens']
notna_index = ~ df_reg[pitch_features].isnull().T.any()
df_reg = df_reg.loc[notna_index]

# max min standardization
for i in pitch_features:
    max_i = df_reg[i].max()
    min_i = df_reg[i].min()
    df_reg[i] = df_reg[i].apply(lambda x: (x - min_i) / (max_i - min_i))

#%% OLS regression (status)
X = df_reg[pitch_features]
X_const = sm.add_constant(X)
Y = df_reg[['status']]
result = sm.OLS(Y, X_const).fit()
print(result.summary())
print(result.summary().as_latex())

# %% logistic regression (status)

model = sm.Logit(Y, X_const)
result = model.fit()

print(result.summary())
print(result.summary().as_latex())

#%% OLS regression (population)
X = df_reg[pitch_features]
X_const = sm.add_constant(X)
Y = df_reg[['population']]
result = sm.OLS(Y, X_const).fit()
print(result.summary())
print(result.summary().as_latex())

#%% OLS regression (posterior success)
X = df_reg[pitch_features]
X_const = sm.add_constant(X)
Y = df_reg[['pos_suc_url']]
result = sm.OLS(Y, X_const).fit()
print(result.summary())
print(result.summary().as_latex())

# %% logistic regression (posterior success)

model = sm.Logit(Y, X_const)
result = model.fit()

print(result.summary())
print(result.summary().as_latex())

#%% OLS regression (pledged amount)
X = df_reg[pitch_features]
X_const = sm.add_constant(X)
Y = df_reg[['done_amount']]
result = sm.OLS(Y, X_const).fit()
print(result.summary())
print(result.summary().as_latex())

#%% OLS regression (goal amount)
X = df_reg[pitch_features]
X_const = sm.add_constant(X)
Y = df_reg[['goal_amount']]
result = sm.OLS(Y, X_const).fit()
print(result.summary())
print(result.summary().as_latex())

# %%
