import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from helpers.eda import *
from helpers.data_prep import *
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("W7/diabetes.csv",na_values={"Glucose": 0,
                                                     "BloodPressure": 0,
                                                     "SkinThickness": 0,
                                                     "Insulin": 0,
                                                     "BMI":0})
df.columns = [col.upper() for col in df.columns]

missing_values_table(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##########################
# OUTLIER
##########################

for col in num_cols:
    print(col,check_outlier(df,col))

for col in num_cols:
    new_df = outlier_thresholds(df, col)

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col,check_outlier(df,col))

##########################
# MISSING VALUE
##########################
msno.matrix(df, figsize=(10,5), fontsize=12, color=(1, 0.38, 0.27));
plt.show()

msno.heatmap(df, cmap="RdYlGn", figsize=(10,5), fontsize=12);
plt.show()

missing_values_table(df)
cols = missing_values_table(df, True)

for i in cols:
    print(f" {i} ".center(50, "*"))
    print(df.groupby("OUTCOME")[i].median())

for col in cols:
    df_ = df[df[col].notnull()]
    df.loc[(df[col].isnull()) & (df["OUTCOME"] == 0), col] = df_.groupby("OUTCOME")[col].median()[0]
    df.loc[(df[col].isnull()) & (df["OUTCOME"] == 1), col] = df_.groupby("OUTCOME")[col].median()[1]

##########################
# FEATURE ENGINEERING
##########################
df.loc[(df['AGE'] <= 30), "NEW_AGE"] = "YOUNG"
df.loc[(df['AGE'] > 30) & (df["AGE"] <= 50), "NEW_AGE"] = "MIDDLE_AGE"
df.loc[(df['AGE'] > 50), "NEW_AGE"] = "OLD"
df.head()

df["INSULIN"].describe()
df.loc[(df['INSULIN'] <= 100), "INSULIN_DEGREE"] = "NORMAL"
df.loc[(df['INSULIN'] >= 100) &(df["INSULIN"] < 126), "INSULIN_DEGREE"] = "AT_RISK"
df.loc[(df['INSULIN'] >= 126), "INSULIN_DEGREE"] = "DIABETES"
df.head()


df.loc[(df["BMI"] < 25 ), "BMI_QUAN"] = "HEALTHY"
df.loc[(df["BMI"] >= 25) &(df["BMI"] < 35) ,"BMI_QUAN"] = "FAT"
df.loc[(df["BMI"] >= 40), "BMI_QUAN"] = "OBESE"

##########################
# ONE-HOT ENCODING
##########################

ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() >= 2) & (col not in ["OUTCOME"])]

df = one_hot_encoder(df, ohe_cols, drop_first=True)

for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])


y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X_train)
y_pred[0:10]
y_train[0:10]

y_prob = log_model.predict_proba(X_train)[:,1]

y_pred = log_model.predict(X_test)
accuracy_score(y_test, y_pred)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# ACCURACY
accuracy_score(y_test, y_pred)

# PRECISION
precision_score(y_test, y_pred)

# RECALL

recall_score(y_test, y_pred)

# F1
f1_score(y_test, y_pred)

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)

print(classification_report(y_test, y_pred))