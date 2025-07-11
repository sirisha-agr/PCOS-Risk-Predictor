from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import pickle
import pandas as pd

df=pd.read_csv('pcos_dataset.csv')
print(df.head(15))

X=df.drop(columns={'PCOS_Diagnosis'})  #features
y=df['PCOS_Diagnosis']  #target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))

print("Random Forest:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=["No PCOS", "PCOS"], cmap="Blues")


with open('pcos_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
print("Model saved successfully as pcos_model.pkl")