from flask import Flask,render_template,request
from mlmodel import *


f1=Flask(__name__)


@f1.route("/")
def home():
    return render_template("page1.html")


@f1.route("/getpredict",methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        credit_policy=request.form['credit_policy']
        purpose=request.form['purpose']
        int_rate=request.form['int_rate']
        installment=request.form['installment'] 
        log_annual_inc=request.form['log_annual_inc']
        dti=request.form['dti']
        fico=request.form['fico']
        days_with_cr_line=request.form['days_with_cr_line']
        revol_bal=request.form['revol_bal']
        revol_util=request.form['revol_util']
        inq_last_6mths=request.form['inq_last_6mths']
        delinq_2yrs=request.form['delinq_2yrs']
        pub_rec=request.form['pub_rec']
        print(credit_policy)
        print(purpose)
        print(int_rate)
        print(installment)
        print(log_annual_inc)
        print(dti)
        print(fico)
        print(days_with_cr_line)
        print(revol_bal)
        print(revol_util)
        print(inq_last_6mths)
        print(delinq_2yrs)
        print(pub_rec)
        
        newobs=np.array([[credit_policy, purpose,int_rate,installment,log_annual_inc,dti, fico, days_with_cr_line,revol_bal,revol_util,inq_last_6mths,delinq_2yrs,pub_rec]],dtype=float)
        print(newobs)
        model=makepredict()
        yp=model.predict(newobs)[0]
        print(yp)


        return render_template("page2.html",data=yp)
    
if __name__=="__main__":
    f1.run(debug=True)