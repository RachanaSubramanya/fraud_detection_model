# importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def train_standard_rfc(x_train_res, y_train_res, x_test):
    """Model 1: Standard Random Forest using SMOTE data."""
    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rfc.fit(x_train_res, y_train_res)
    return rfc.predict(x_test)


def train_optimized_rfc(x_train_res, y_train_res, x_test):
    """Model 2: Optimized Random Forest using SMOTE data."""
    # Optimized hyperparameters were obtained from GridSearchCV which was run separately and not included in this script to save time.
    rfc_opt = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=2, 
                                     class_weight='balanced', random_state=42, n_jobs=-1)
    rfc_opt.fit(x_train_res, y_train_res)
    return rfc_opt.predict(x_test)


def train_xgboost(x_train, y_train, x_test):
    """Model 3: XGBoost using original data and scale_pos_weight."""
    scale_factor = sum(y_train == 0) / sum(y_train == 1)
    xgb_model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, 
                              scale_pos_weight=scale_factor, random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb_model.fit(x_train, y_train)
    return xgb_model.predict(x_test)


def train_balanced_rfc(x_train, y_train, x_test):
    """Model 4: Balanced Random Forest using internal down-sampling."""
    brfc = BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    brfc.fit(x_train, y_train) 
    return brfc.predict(x_test)


def train_logistic_regression(x_train_scaled, y_train_res, x_test_scaled):
    """Model 5: Logistic Regression Baseline using scaled SMOTE data."""
    model_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)  
    model_lr.fit(x_train_scaled, y_train_res)                                                
    return model_lr.predict(x_test_scaled)                                               


def train_linear_svc(x_train_scaled, y_train_res, x_test_scaled):
    """Model 6: Linear Support Vector Machine using scaled SMOTE data."""
    model_svm = LinearSVC(class_weight='balanced', dual=False, random_state=42)            
    model_svm.fit(x_train_scaled, y_train_res)                                               
    return model_svm.predict(x_test_scaled)                                              


def train_knn(x_train_scaled, y_train_res, x_test_scaled):
    """Model 7: K-Nearest Neighbors using scaled SMOTE data."""
    model_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)                             
    model_knn.fit(x_train_scaled, y_train_res)                                               
    return model_knn.predict(x_test_scaled)