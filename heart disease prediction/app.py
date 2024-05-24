# from numpy import less
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./heart.csv")
df2 = df.drop(labels=['target'],axis=1)
X = df2.iloc[:, :]
y = df['target'].iloc[:]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=20)

#feature scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


app = Flask(__name__)

model = joblib.load("Heart_Model.pkl")

@app.route("/", methods=['GET','POST'])
def FrontPage():
    # return f"<h1>{model.predict(sc_X.transform([[45,1,3,110,264,0,1,132,0,1.2,1,0,3]]))}</h1>"
    # return f"<h1>{model.predict(sc_X.transform([[51,1,2,110,175,0,1,123,0,0.6,2,0,2]]))}</h1>"
    # return render_template("me.html")
    return render_template("index.html")
    # return render_template("http://192.168.218.212:5500/frontend/src/index.html")

@app.route("/reply", methods=['GET', 'POST'])
def FrontPag():
    data = request.json
    
    # Accessing values from request.json
    age = data.get('Age')
    sex = data.get('Sex')
    cp = data.get('Cp')
    bp = data.get('Bp')
    chol = data.get('Chol')
    fbs = data.get('Fbs')
    restecg = data.get('Restecg')
    thalach = data.get('Thalach')
    exang = data.get('Exang')
    oldpeak = data.get('Oldpeak')
    slope = data.get('Slope')
    ca = data.get('Ca')
    thal = data.get('Thal')
    
    # Printing the values for debugging purposes
    print("Age:", age)
    print("Sex:", sex)
    print("Cp:", cp)
    print("Bp:", bp)
    print("Chol:", chol)
    print("Fbs:", fbs)
    print("Restecg:", restecg)
    print("Thalach:", thalach)
    print("Exang:", exang)
    print("Oldpeak:", oldpeak)
    print("Slope:", slope)
    print("Ca:", ca)
    print("Thal:", thal)

    # Rest of your code
    result = model.predict(sc_X.transform([[int(age), int(sex), int(cp), int(bp), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]]))
    print(result)
    return str(result[0])


if __name__ == "__main__":
    app.run(debug=True,port=8000)