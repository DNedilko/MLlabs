
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from subprocess import check_call
from sklearn import metrics, preprocessing
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from yellowbrick.classifier.rocauc import roc_auc, roc_curve
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('WQ-R.csv',sep=';')

r,c = df.shape
print(f'У поданому файлі {r} записів та {c} полів. \nПерші 10 записів файлу:\n', df.head(10).to_string())


le = preprocessing.LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
train, test = train_test_split(df, test_size=0.2)
test_l, test_f = test['quality'], test.drop('quality',axis=1)

clfE = DecisionTreeClassifier(criterion = 'entropy',max_depth = 5, random_state = 0)
clfE.fit(train.drop('quality',axis=1),train['quality'])

score_E = clfE.predict(test_f)

clfG = DecisionTreeClassifier(max_depth = 5, random_state = 0)
clfG.fit(train.drop('quality',axis=1),train['quality'])
score_G = clfG.predict(test_f)

fn=df.drop('quality',axis=1).columns.values.tolist()
cn=df['quality'].unique()
tree.export_graphviz(clfG,out_file="treeG.dot",feature_names = fn,rounded=True, filled = True)
check_call(['dot','-Tpng','treeG.dot','-o','treeG.png'])

tree.export_graphviz(clfE,out_file="treeE.dot",feature_names = fn,rounded=True, filled = True)
check_call(['dot','-Tpng','treeE.dot','-o','treeE.png'])

'''Обчислити класифікаційні метрики збудованої моделі для тренувальної
та тестової вибірки. Представити результати роботи моделі на тестовій
вибірці графічно. Порівняти результати, отриманні при застосуванні
різних критеріїв розщеплення: інформаційний приріст на основі
ентропії чи неоднорідності Джині.'''


CME = metrics.confusion_matrix(test_l, score_E)
ax = sns.heatmap(CME, annot=True, cmap = 'Blues')

ax.set_title('Confusion Matrix')
ax.set_xlabel('\n  Predicted values')
ax.set_ylabel('\n  Actual values')

plt.show()


# precision, recall, thresholds = precision_recall_curve(test_l, score_E)


gini_report = metrics.classification_report(test_l, score_G,output_dict=True)
entropy_report = metrics.classification_report(test_l, score_E,output_dict=True)
fig = sns.heatmap(pd.DataFrame(gini_report).iloc[:-1, :].T, annot=True)
plt.show()
fig.get_figure().clf()
fig = sns.heatmap(pd.DataFrame(entropy_report).iloc[:-1, :].T, annot=True)
plt.show()
fig.get_figure().clf()
'''З’ясувати вплив глибини дерева та мінімальної кількості елементів в
листі дерева на результати класифікації. Результати представити
графічно.'''
dept = np.array(range(4,14))
samples = np.array(range(5,200,20))

dic = []
for i in dept:
    for s in samples:
        clf = DecisionTreeClassifier(criterion='entropy',max_depth=i,min_samples_split=s).fit(train.drop('quality',axis=1),train['quality']).predict(test_f)
        dic.append(metrics.accuracy_score(test_l,clf))

k = np.array(dic).reshape(10,10)

acc = pd.DataFrame(k, index=samples, columns=dept)
print(acc)
plt.plot(dept, k[1,:])
plt.title("Залежність точності класифікації від глибини дерева")
plt.show()
plt.plot(samples,k[:,0])
plt.title("Залежність точності класифікації від мінімальної кількості елементів")
plt.show()

''' Навести стовпчикову діаграму важливості атрибутів, які
використовувалися для класифікації (див. feature_importances_).
Пояснити, яким чином – на Вашу думку – цю важливість можна
підрахувати.'''


importance = pd.DataFrame({'feature': train.drop('quality',axis=1).columns, 'importance': np.round(clfG.feature_importances_,3)})
importance.sort_values('importance', ascending = False, inplace=True)
plt.figure(figsize=(9,9))
importance.plot.bar("feature",'importance',rot=30)
plt.title("Діаграма важливості артибутів")
plt.show()

#
# # y=preprocessing.label_binarize(test_l, classes=[0,1,2,3,4])
# Gsc = clfE.predict_proba(test_f)
# Esc = clfE.predict_proba(test_f)
#
# false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(test_l, Gsc)
# false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(test_l, Esc)
# try:
#     print('roc_auc_score для дерева з критерієм ентропія: ', roc_auc_score(y, Esc))
#     print('roc_auc_score для дерева з критерієм неоднорідності Джинні: ', roc_auc_score(y, Gsc))
# except ValueError:
#     pass
#
# plt.subplots(1, figsize=(10, 10))
# plt.title('DecisionTree - Entropy')
# plt.plot(false_positive_rate1, true_positive_rate1)
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
#
# plt.subplots(1, figsize=(10, 10))
# plt.title('DecisionTree - Gini index')
# plt.plot(false_positive_rate2, true_positive_rate2)
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()