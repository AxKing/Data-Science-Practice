# Esemble Methods- method of creating multiple models and then combining them together to make one model

# Starts with a stump

# Build our Own Grid-Search for Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
print(dir(GradientBoostingClassifier))

from sklearn.metrics import prcision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

x_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)

def train_GB(est, max_depth, lr):
	gb = GradientBoostingClassifier(n_estimators=est, max_depth=max_depth, learning_rate=lr)
	gb_model = gb.fit(X_train, y_train)
	y_pred = gb_model.predict(X_test)	
	precision, recall, fscore, support = score(y_test, y_pred, pos_label = 'spam', average='binary')
	print('EST: {} / Depth: {} / LR: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
		est, max_depth, lr, round(precision, 3), round(recall, 3), round(y_pred==y_test.sum() / len(y_pred), 3)))

for n_est in [50, 100, 150]:
	for max_depth in [3, 7, 11, 15]:
		for lr in [0.01, 0.1, 1]:
			traing_GB(n_est, max_depth, lr)


 gb = GradientBoostingClassifier()
 param = {
 'n_estimators': [100, 150],
 'max_depth': [7,11,15],
 'learning_rate': [0.1]
 }

gs = GridSearchCV(gb, param, cv=5, n_jobs=-1)

cv_fit = gs.fit(X_tfidf_feat, data['label'])
pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]





