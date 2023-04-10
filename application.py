from flask import Flask,request,render_template
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
import pickle

application = Flask(__name__)
app=application

model=pickle.load(open("models/model3.pkl","rb"))
decoder=pickle.load(open("models/time.pkl","rb"))

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/calculate_tip",methods=["POST"])
def calculate_tip():
    if request.method=="POST":
        tip0=float(request.form.get("total_bill"))
        tip1=float(request.form.get("total_tip"))
        tip2=request.form.get("time_of_event")

        tip3=decoder.transform([tip2])[0]

        result=model.predict([[tip0,tip1,tip3]])


        return render_template("index.html",tip_amount=result[0])



if __name__=="__main__":
    app.run(host="0.0.0.0")
