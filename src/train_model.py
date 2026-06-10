import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from lightgbm import LGBMClassifier

from sklearn.model_selection import (train_test_split,cross_val_score)

from sklearn.metrics import (roc_auc_score,precision_score,recall_score,f1_score,classification_report,precision_recall_curve)

def train_model(df):
    X = df.drop(['TARGET'],axis=1)
    y = df['TARGET']
    print("\nFinal Shape:")
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    print("\nTrain Shape:")
    print(X_train.shape)

    print("\nTest Shape:")
    print(X_test.shape)

    mlflow.end_run()

    mlflow.set_experiment("Home_Credit_Project")
    with mlflow.start_run():
        model = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            metric='auc',
            n_estimators=5000,
            learning_rate=0.007,
            num_leaves=32,
            max_depth=6,
            min_child_samples=180,
            subsample=0.8,
            subsample_freq=1, 
            colsample_bytree=0.65,
            reg_alpha=8, 
            reg_lambda=18,
            scale_pos_weight=11,
            random_state=42, 
            n_jobs=-1,
            verbose=-1
        )
        
        # CROSS VALIDATION

        scores = cross_val_score(model,X_train,y_train,cv=5,scoring='roc_auc',n_jobs=-1)

        print("\nCV ROC-AUC Scores:")
        print(scores)

        print("\nAverage ROC-AUC:")
        print(scores.mean())

    model.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric='auc')
    y_prob = model.predict_proba(X_test)[:, 1]

    ## Threshold tuning
    precisions, recalls, thresholds = precision_recall_curve(y_test,y_prob)
    f1_scores = (2 * precisions * recalls /(precisions + recalls + 1e-8))
    best_idx = np.argmax(f1_scores)
    # best_threshold = thresholds[best_idx]
    best_threshold = 0.45

    print("\nBest Threshold:")
    print(best_threshold)
    y_pred = (y_prob >= best_threshold).astype(int)

    #Evaluation
    auc = roc_auc_score(y_test,y_prob)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    print("\nROC-AUC Score:")
    print(auc)
    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))
    print("\nPrecision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    mlflow.log_metric("roc_auc",auc)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)
    mlflow.sklearn.log_model(model,"lightgbm_model")
    print("\nMLflow Run Completed")

    joblib.dump(model,"models/model.pkl") 
    # joblib.dump(X.columns.tolist(),"models/final_model.pkl")
    print("Model Saved")

    train_model(df)



