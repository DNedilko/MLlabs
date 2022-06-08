import pandas as pd
from sklearn import metrics as m
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def information(dataset_name):
    df = pd.read_csv(dataset_name, sep=";")
    r, c = df.shape
    print(f'Файл складається з {r} рядків та {c}.\n{"*"*70} \nТип полів: \n{df.dtypes} \n{"*"*70}\n'
          f'Перші 10 записів в {dataset_name}: \n {df.head(10).to_string()}')
    return df






def main(df):
    accuracy_scores = []
    label_name = "HighQuality"
    df[label_name] = df["quality"].apply(lambda x: 1 if x>=6 else 0)
    print(f'\nВидозмінений датафрем набуває такого вигляду {df.head(10).to_string()}\n')
    df = df.drop("quality", axis = 1)
    Y = df[label_name]
    X = df.drop(label_name, axis=1)
    feature_names = X.columns
    #print(feature_names)
    cv = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    for train_index, test_index in cv.split(X):
        x_train = df.loc[train_index, feature_names]
        x_test = df.loc[test_index, feature_names]
        y_train = df.loc[train_index, label_name]
        y_test = df.loc[test_index, label_name]
        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        y_train_pred = reg.predict(x_train)

        # acc = m.accuracy_score(y_test, y_pred)
        # print(acc)
        CME = m.confusion_matrix(y_test, y_pred)
        # print(CME)
        ax = sns.heatmap(CME, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('\n  Predicted values')
        ax.set_ylabel('\n  Actual values')
        plt.show()

        report_train = m.classification_report(y_train, y_train_pred,output_dict=True)
        report_test = m.classification_report(y_test, y_pred, output_dict=True)
        # fig, (ax1,ax2) = plt.subplot(2,1,sharex=True, figsize=(12,12))
        fig = sns.heatmap(pd.DataFrame(report_test).iloc[:-1, :].T, annot=True)
        fig.set_title("Classification reports: Test")
        plt.show()
        fig = sns.heatmap(pd.DataFrame(report_train).iloc[:-1, :].T, annot = True)
        fig.set_title("Classification reports: Train")
        plt.show()
        #print(f'Classification report:\n  Training set:\n{report_train}\n   Test set:{report_test}')


        y_pred_prob = reg.predict_proba(x_test)[::,1]
        fpr, tpr, _ = m.roc_curve(y_test, y_pred_prob)
        auc = m.roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr,label="AUC="+str(auc))
        plt.title("ROC крива для тестової вибірки")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()

        # fig, ax = plt.subplots()
        # ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
        # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        # ax.set_xlabel('Predicted')
        # ax.set_ylabel('Actual')
        # plt.show()
        mae = m.mean_absolute_error(y_test, y_pred)
        mse = m.mean_squared_error(y_test, y_pred)
        r2 = m.r2_score(y_test, y_pred)

        print("The model performance for testing set")
        print("--------------------------------------")
        print('MAE is {}'.format(round(mae,5)))
        print('MSE is {}'.format(round(mse,5)))
        print('R2 score is {}'.format(r2))

        accuracy_train = []
        accuracy_test = []
        x =[]
        for i in range(5,5001,45):
            reg9 = LogisticRegression(solver='lbfgs', max_iter=i)
            reg9.fit(x_train, y_train)
            y_pred = reg9.predict(x_test)
            y_train_pred = reg9.predict(x_train)
            accuracy_train.append(m.accuracy_score(y_train, y_train_pred))
            accuracy_test.append(m.accuracy_score(y_test, y_pred))
            x.append(i)
        plt.plot(x, accuracy_train, color='blue', label='Train accuracy')
        plt.plot(x,accuracy_test, color='yellow', label='Test accuracy')
        plt.legend()
        plt.title("Залежність точності передбачення від кількості ітерацій")
        plt.show()

        imptance = pd.DataFrame({'feature': feature_names, 'imp': reg.coef_[0]})
        imptance = imptance.sort_values(by='imp', ascending=False)

        for i, j in enumerate(imptance['imp']):
            print(f'Feature: {imptance["feature"][i]}, Score: {j}')


        plt.bar([X for X in imptance['feature']], imptance['imp'])
        plt.xticks(rotation=45)
        plt.title('Feature importance')
        plt.show()

        new_train_x = x_train[imptance.head(5)['feature'].to_list()]
        new_test_x = x_test[imptance.head(5)['feature'].to_list()]
        reg = LogisticRegression(solver='lbfgs', max_iter=300)
        reg.fit(new_train_x, y_train)
        y_pred_new = reg.predict(new_test_x)
        report_test = m.classification_report(y_test, y_pred_new, output_dict=True)
        # fig, (ax1,ax2) = plt.subplot(2,1,sharex=True, figsize=(12,12))
        fig = sns.heatmap(pd.DataFrame(report_test).iloc[:-1, :].T, annot=True)
        fig.set_title("Тренування на основі перший 5ти по важливості компонентів")
        plt.show()


        # importance = pd.DataFrame({'feature': feature_names, 'importance': np.round(reg.feature_importances_, 3)})
        # importance.sort_values('importance', ascending=False, inplace=True)
        # plt.figure(figsize=(9, 9))
        # importance.plot.bar("feature", 'importance', rot=30)
        # plt.title("Діаграма важливості артибутів")
        #



if __name__ == "__main__":
    file_name = "WQ-R.csv"
    df = information(file_name)
    main(df)







