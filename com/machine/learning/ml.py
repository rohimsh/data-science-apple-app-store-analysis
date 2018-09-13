import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Gaussian Process": GaussianProcessClassifier()
}



def main():
    # loading our data as a panda
    training_df = pd.read_csv('training_data.csv', delimiter=",")
    test_df = pd.read_csv('test_data.csv', delimiter=",")
    training_features = training_df.drop(['app_rating', 'app_name'], axis=1)
    test_features = test_df.drop(['app_rating', 'app_name'], axis=1)
    lab_enc = preprocessing.LabelEncoder()
    label_target_training = training_df['app_rating']
    label_target_test = test_df['app_rating']
    label_target_encoded = lab_enc.fit_transform(label_target_training)
    label_test_encoded = lab_enc.fit_transform(label_target_test)

    for classifier_name, classifier in list(dict_classifiers.items()):
        print("Classifier: " + classifier_name)
        classifier.fit(training_features, label_target_encoded)
        training_score = classifier.score(training_features, label_target_encoded)
        print("Training Score: " + str(training_score))
        test_score = classifier.score(test_features, label_test_encoded)
        print("Test Score: " + str(test_score))


if __name__ == '__main__':
    main()