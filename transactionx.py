from flask import *
from sklearn.metrics import f1_score, recall_score, precision_score

app = Flask(__name__)
from DBConnection import *
app.secret_key="987654123456"
staticpath="C:\\Users\\lenshif\\Desktop\TRANSACTIONX\\transactionx\\static\\"
@app.route('/')
def hello_world():
    return render_template('loginindex.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['textfield']
    password=request.form['textfield2']
    qry="SELECT * FROM login WHERE user_name='"+username+"' AND PASSWORD='"+password+"'"
    db=Db()
    res=db.selectOne(qry)
    if res is None:
        return "<script>alert('invalid username or password');window.location='/'</script>"
    else:
        if res['type']=='admin':
            session['lid']=res['login_id']
            return redirect('/ahome')
        elif res['type']=='user':
            session['lid']=res['login_id']
            return redirect('/uhome')
        else:
            "<script>alert('invalid username or password');window.location='/'</script>"

@app.route('/n')
def n():
    return render_template('users/n.html')

@app.route('/uhome')
def uhome():
    return render_template("users/user_index.html")
@app.route('/ahome')
def ahome():
    return render_template("admin/admin_index.html")




@app.route('/change_password')
def change_password():
    return render_template("admin/change password.html")

@app.route('/changepasswordpost', methods=['POST'])
def changepasswordpost():
    currentpassword=request.form['textfield']
    newpassword=request.form['textfield2']
    confirmpassword=request.form['textfield3']
    qry="SELECT * FROM login WHERE PASSWORD='"+currentpassword+"' AND login_id='"+str(session['lid'])+"'"
    db=Db()
    res=db.selectOne(qry)
    if res is None:
        return "<script>alert('invalid password');window.location='/ahome'</script>"
    elif(newpassword!=confirmpassword):
        return "<script>alert('password mismatch');window.location='/ahome'</script>"
    else:
        qry="UPDATE login SET PASSWORD='"+newpassword+"' WHERE PASSWORD='"+currentpassword+"' AND login_id='"+str(session['lid'])+"'"
        db=Db()
        db.update(qry)
        return "<script>alert('password changed');window.location='/'</script>"







@app.route('/view_feedback')
def view_feedback():
    qry="SELECT * FROM feedback JOIN USER ON feedback.lid=user.lid"
    db=Db()
    res=db.select(qry)
    return render_template("admin/view feedback.html",data=res)

@app.route('/view_users')
def view_users():
    qry="SELECT * FROM USER"
    db=Db()
    res=db.select(qry)
    return render_template("admin/view users.html",data=res)

@app.route('/change_password_user')
def change_password_user():
    return render_template("users/change password.html")

@app.route('/change_passsowrd_user_post', methods=['POST'])
def change_passsowrd_user_post():
    currentpassword=request.form['textfield']
    newpassword = request.form['textfield2']
    confirmpassword = request.form['textfield3']
    qry = "SELECT * FROM login WHERE PASSWORD='" + currentpassword + "' AND login_id='" + str(session['lid']) + "'"
    db = Db()
    res = db.selectOne(qry)
    if res is None:
        return "<script>alert('invalid password');window.location='/uhome'</script>"
    elif (newpassword != confirmpassword):
        return "<script>alert('password mismatch');window.location='/uhome'</script>"
    else:
        qry = "UPDATE login SET PASSWORD='" + newpassword + "' WHERE PASSWORD='" + currentpassword + "' AND login_id='" + str(session['lid']) + "'"
        db = Db()
        db.update(qry)
        return "<script>alert('password changed');window.location='/'</script>"



@app.route('/edit_profile')
def edit_profile():
    qry = "SELECT * FROM USER WHERE lid='" + str(session['lid']) + "'"
    db = Db()
    res = db.selectOne(qry)
    return render_template("users/edit profile.html",data=res)

@app.route('/editprofilepost', methods=['POST'])
def editprofilepost():
    name=request.form['textfield']
    gender=request.form['RadioGroup1']
    place=request.form['textfield2']
    post=request.form['textfield3']
    pin=request.form['textfield4']
    country=request.form['textfield5']
    email=request.form['email']
    if 'fileField' in request.files:
        photo=request.files['fileField']
        if photo.filename!="":
            from datetime import datetime
            date = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
            photo.save(staticpath + "user\\" + date)
            path = "/static/user/" + date
            qry="UPDATE USER SET NAME='"+name+"',gender='"+gender+"',photo='"+path+"',place='"+place+"',post='"+post+"',pin='"+pin+"',country='"+country+"',email='"+email+"' WHERE lid='"+str(session['lid'])+"'"
            db=Db()
            db.update(qry)
            return "ok"
        else:
            qry="UPDATE USER SET NAME='"+name+"',gender='"+gender+"',place='"+place+"',post='"+post+"',pin='"+pin+"',country='"+country+"',email='"+email+"' WHERE lid='"+str(session['lid'])+"'"
            db = Db()
            db.update(qry)
            return "ok"
    else:
        qry = "UPDATE USER SET NAME='" +name+"',gender='" +gender+"',place='" +place+"',post='" +post+"',pin='" +pin+"',country='" +country+"',email='" +email+"' WHERE lid='"+str(session['lid'])+"'"
        db = Db()
        db.update(qry)
        return "ok"







@app.route('/send_feedback')
def send_feedback():
    return render_template("users/send feedback.html")



@app.route('/send_feedback_post', methods=['POST'])
def send_feedback_post():
    feedback=request.form['textfield']
    qry="INSERT INTO feedback (lid,feedback,DATE) VALUES ('"+str(session['lid'])+"','"+feedback+"',CURDATE())"
    db=Db()
    db.insert(qry)

    return "ok"

@app.route('/signup_index')
def signup_index():
    return render_template('regindex.html')


@app.route('/sign_up')
def sign_up():
    return render_template("regindex.html")

@app.route('/sign_up_post',methods=['post'])
def sign_up_post():
    name=request.form['textfield']
    print (name)
    gender=request.form['RadioGroup1']
    print (gender)
    photo=request.files['fileField']

    from datetime import datetime
    date=datetime.now().strftime("%Y%m%d%H%M%S")+".jpg"
    photo.save(staticpath+"user\\"+date)
    path="/static/user/"+date
    place=request.form['textfield2']
    post=request.form['textfield3']
    pin=request.form['textfield4']
    country=request.form['textfield5']
    email=request.form['Email']
    password=request.form['textfield6']
    qry="INSERT INTO login(user_name,PASSWORD,TYPE) VALUES('"+email+"','"+password+"','user')"
    db=Db()
    lid=db.insert(qry)
    qry2="INSERT INTO USER(lid,NAME,gender,photo,place,post,pin,country,email) VALUES('"+str(lid)+"','"+name+"','"+gender+"','"+path+"','"+place+"','"+post+"','"+pin+"','"+country+"','"+email+"')"
    db=Db()
    db.insert(qry2)
    return redirect('/')



@app.route('/view_profile')
def view_profile():
    qry="SELECT * FROM USER WHERE lid='"+str(session['lid'])+"'"
    db=Db()
    res=db.selectOne(qry)
    return render_template("users/view profile.html",data=res)

@app.route('/view_feedback_user')
def view_feedback_user():
    qry="select * from feedback where lid='"+str(session['lid'])+"'"
    db=Db()
    res=db.select(qry)
    return render_template("users/view feedback user.html",data=res)

@app.route('/delete_feedback/<id>')
def delete_feedback(id):
    qry="DELETE FROM `feedback` WHERE uid='"+id+"'"
    db=Db()
    db.delete(qry)
    return redirect('/view_feedback_user')


@app.route('/viewresult')
def viewresult():
    from algo import fake
    res=fake()
    return render_template("admin/viewresult.html",data=res)




@app.route('/viewresultrf')
def viewresultrf():
    return render_template("users/resultrf.html")



@app.route('/view_result_rf_post', methods=['POST'])
def view_result_rf_post():

    file=request.files['file']
    from datetime import datetime
    date=datetime.now().strftime("%Y%m%d%H%M%S")+file.filename
    file.save("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\"+date)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    mydata = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)


    X = mydata.values[:1000, 0:30]
    y =mydata.values[:1000, 30]


    print(y)

    # return "ok"

    # X = X[:25000]
    # y = y[:25000]
    print("-------------------%----------")
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("split")
    from sklearn.ensemble import RandomForestClassifier
    print("import")
    regressor = RandomForestClassifier(n_estimators=1000, random_state=42)
    print("regr")
    regressor.fit(X_train, y_train)

    print("fit")
    y_pred = regressor.predict(X_test)
    print(y_pred, "value")
    print(X_test, "-----------------------------------------")
    for i in X_test:
        print(i)

    from sklearn.metrics import confusion_matrix, accuracy_score
    c=confusion_matrix(y_pred,y_test)
    acc = accuracy_score(y_pred, y_test)
    print(acc,"hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")




    ##########################################fportion
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


    acclk = accuracy_score(y_pred, y_test)
    f1lik = f1_score(y_pred, y_test)
    recalllik = recall_score(y_pred, y_test)
    preclik = precision_score(y_pred, y_test)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_pred, y_test, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cmd.plot()

    cmd.figure_.savefig('C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\conf_mat.png', dpi=300)

    import matplotlib.pyplot as plt
    plt.cla()
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(y_pred, y_test)
    auc = metrics.roc_auc_score(y_pred, y_test)



    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)


    plt.savefig("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\aba.jpg")
    return render_template("users/viewresultrf.html", data=y_pred,test=X_test ,l =len(y_pred),cm=c,acu=acc,preclik=preclik,acclk=acclk,f1lik=f1lik,recalllik=recalllik)

@app.route('/viewresultnb')
def viewresultnb():
    return render_template("users/resultnb.html")


@app.route('/view_result_nb_post', methods=['POST'])
def view_result_nb_post():
    file = request.files['file']
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d%H%M%S") + file.filename
    file.save("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    mydata = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)



    X = mydata.values[:1000, 0:30]
    y =mydata.values[:1000, 30]

    print(y)

    # return "ok"

    # X = X[:25000]
    # y = y[:25000]
    print("-------------------%----------")
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("split")
    from sklearn.naive_bayes import GaussianNB
    print("import")
    regressor = GaussianNB()
    print("regr")
    regressor.fit(X_train, y_train)

    print("fit")
    y_pred = regressor.predict(X_test)
    print(y_pred, "value")
    print(X_test,"-----------------------------------------")


    for i in X_test:
        print(i)


    from sklearn.metrics import  confusion_matrix,accuracy_score
    c=confusion_matrix(y_pred,y_test)
    acc =accuracy_score(y_pred, y_test)
    print(acc, "hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

    acclk = accuracy_score(y_test, y_pred)
    f1lik = f1_score(y_test, y_pred)
    recalllik = recall_score(y_test, y_pred)
    preclik = precision_score(y_test, y_pred)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cmd.plot()

    cmd.figure_.savefig('C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\conf_mat.png', dpi=300)

    import matplotlib.pyplot as plt
    plt.cla()
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    y_predictedlk = y_pred
    X_testlk = X_test

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    # plt.show()



    plt.savefig("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\aba.jpg")

    y_predictedlk = y_pred
    X_testlk = X_test

    llk = len(X_testlk)

    conf_matlk = c
    acclk = accuracy_score(y_test, y_pred)
    print(X_testlk)




    return render_template("users/viewresultnb.html",  l=len(y_predictedlk), y_predictedlk=y_predictedlk,
                       X_testlk=X_testlk, acclk=acclk, f1lik=f1lik, recalllik=recalllik, preclik=preclik,
                       data=y_predictedlk, test=X_testlk, cm=conf_matlk, acu=acclk)


@app.route('/viewresultdt')
def viewresultdt():
    return render_template("users/resultdt.html")


@app.route('/viewresultdt_post', methods=['POST'])
def viewresultdt_post():
    file = request.files['file']
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d%H%M%S") + file.filename
    file.save("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    mydata = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    X = mydata.values[:1000, 0:30]
    y = mydata.values[:1000, 30]

    print(y)

    # return "ok"

    # X = X[:25000]
    # y = y[:25000]
    print("-------------------%----------")
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("split")
    from sklearn.tree import DecisionTreeClassifier
    print("import")
    regressor = DecisionTreeClassifier()
    print("regr")
    regressor.fit(X_train, y_train)

    print("fit")
    y_pred = regressor.predict(X_test)
    print(y_pred, "value")
    print(X_test, "-----------------------------------------")

    for i in X_test:
        print(i)

    from sklearn.metrics import confusion_matrix, accuracy_score
    c = confusion_matrix(y_pred, y_test)
    acc = accuracy_score(y_pred, y_test)
    print(acc, "hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

    acclk = accuracy_score(y_test, y_pred)
    f1lik = f1_score(y_test, y_pred)
    recalllik = recall_score(y_test, y_pred)
    preclik = precision_score(y_test, y_pred)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cmd.plot()

    cmd.figure_.savefig('C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\conf_mat.png', dpi=300)

    import matplotlib.pyplot as plt
    plt.cla()
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    y_predictedlk = y_pred
    X_testlk = X_test

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    # plt.show()



    plt.savefig("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\aba.jpg")

    y_predictedlk = y_pred
    X_testlk = X_test

    llk = len(X_testlk)

    conf_matlk = c
    acclk = accuracy_score(y_test, y_pred)
    print(X_testlk)


    return render_template("users/viewresultdt.html", l=len(y_pred),acclk=acclk,f1lik=f1lik, recalllik=recalllik,preclik=preclik, data=y_predictedlk, test=X_test, cm=conf_matlk, acu=acclk)


@app.route('/viewresultsvm')
def viewresultsvm():
    return render_template("users/resultsvm.html")


@app.route('/viewresultsvm_post', methods=['POST'])
def viewresultsvm_post():
    file = request.files['file']
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d%H%M%S") + file.filename
    file.save("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    mydata = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    X = mydata.values[:1000, 0:30]
    y = mydata.values[:1000, 30]

    print(y)

    # return "ok"

    # X = X[:25000]
    # y = y[:25000]
    print("-------------------%----------")
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("split")
    from sklearn.svm import SVC
    print("import")
    regressor = SVC()

    print("regr")
    regressor.fit(X_train, y_train)

    print("fit")
    y_pred = regressor.predict(X_test)
    print(y_pred, "value")
    print(X_test, "-----------------------------------------")

    for i in X_test:
        print(i)

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import confusion_matrix, accuracy_score
    c = confusion_matrix(y_pred, y_test)
    acc = accuracy_score(y_pred, y_test)
    print(acc, "hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

    acclk = accuracy_score(y_test, y_pred)
    f1lik = f1_score(y_test, y_pred)
    recalllik = recall_score(y_test, y_pred)
    preclik = precision_score(y_test, y_pred)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cmd.plot()

    cmd.figure_.savefig('C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\conf_mat.png', dpi=300)

    import matplotlib.pyplot as plt
    plt.cla()
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    y_predictedlk = y_pred
    X_testlk = X_test

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    # plt.show()



    plt.savefig("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\aba.jpg")

    y_predictedlk = y_pred
    X_testlk = X_test

    llk = len(X_testlk)

    conf_matlk = c
    acclk = accuracy_score(y_test, y_pred)
    print(X_testlk)
    return render_template("users/viewresultsvm.html", l=len(y_pred),acclk=acclk,f1lik=f1lik, recalllik=recalllik,preclik=preclik, data=y_predictedlk, test=X_test, cm=conf_matlk, acu=acclk)


@app.route('/viewresultmlp')
def viewresultmlp():
    return render_template("users/resultmlp.html")


@app.route('/viewresultmlp_post', methods=['POST'])
def viewresultmlp_post():
    file = request.files['file']
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d%H%M%S") + file.filename
    file.save("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    mydata = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    X = mydata.values[:1000, 0:30]
    y = mydata.values[:1000, 30]

    print(y)

    # return "ok"

    # X = X[:25000]
    # y = y[:25000]
    print("-------------------%----------")
    print(y)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("split")
    from sklearn.neural_network import MLPClassifier
    print("import")
    regressor = MLPClassifier()

    print("regr")
    regressor.fit(X_train, y_train)

    print("fit")
    y_pred = regressor.predict(X_test)
    print(y_pred, "value")
    print(X_test, "-----------------------------------------")

    for i in X_test:
        print(i)

    from sklearn.metrics import confusion_matrix, accuracy_score
    c = confusion_matrix(y_pred, y_test)
    acc = accuracy_score(y_pred, y_test)
    print(acc, "hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

    acclk = accuracy_score(y_test, y_pred)
    f1lik = f1_score(y_test, y_pred)
    recalllik = recall_score(y_test, y_pred)
    preclik = precision_score(y_test, y_pred)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cmd.plot()

    cmd.figure_.savefig('C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\conf_mat.png', dpi=300)

    import matplotlib.pyplot as plt
    plt.cla()
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    y_predictedlk = y_pred
    X_testlk = X_test

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    # plt.show()



    plt.savefig("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\aba.jpg")

    y_predictedlk = y_pred
    X_testlk = X_test

    llk = len(X_testlk)

    conf_matlk = c
    acclk = accuracy_score(y_test, y_pred)
    print(X_testlk)
    return render_template("users/viewresultmlp.html",  l=len(y_pred),acclk=acclk,f1lik=f1lik, recalllik=recalllik,preclik=preclik, data=y_predictedlk, test=X_test, cm=conf_matlk, acu=acclk)



#lstm


@app.route('/viewresultlstm')
def viewresultlstm():
    return render_template("users/resultlstm.html")


@app.route('/viewresultlstm_post', methods=['POST'])
def viewresultlstm_post():
    file = request.files['file']
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d%H%M%S") + file.filename
    file.save("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date)

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import LSTM, Dense,Dropout
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
        ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from sklearn import metrics

    # Load the data
    data = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\predict\\" + date,
                       na_filter=True)

    # Split the data into input features and labels
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Scale the input features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input features to be compatible with LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(1, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=8))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Print the evaluation metrics

    for i in y_pred:
        print(i)

    acclk = accuracy_score(y_test, y_pred)
    f1lik = f1_score(y_test, y_pred)
    recalllik = recall_score(y_test, y_pred)
    preclik = precision_score(y_test, y_pred)


    cm = confusion_matrix(y_test, y_pred, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cmd.plot()

    cmd.figure_.savefig('C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\conf_mat.png', dpi=300)

    import matplotlib.pyplot as plt
    plt.cla()
    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    y_predictedlk = y_pred
    X_testlk = X_test

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\static\\aba.jpg")

    llk = len(y_pred)

    conf_matlk = cm
    acclk = accuracy_score(y_test, y_pred)
    print(X_testlk)

    print("hellojdkfhsdkjhfkj hskjdhfkj hsdkj", X_test)
    return render_template("users/viewresultlstm.html", l=len(y_predictedlk), y_predictedlk=y_predictedlk,
                           X_testlk=X_testlk, acclk=acclk, f1lik=f1lik, recalllik=recalllik, preclik=preclik,
                           data=y_predictedlk, test=X_testlk, cm=conf_matlk, acu=acclk)


if __name__ == '__main__':
    app.run(debug=True)
