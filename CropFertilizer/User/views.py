from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
# Create your views here.
def home(request):
    return render(request,"index.html")

def login(request):
    if request.method == "POST":
        uname=request.POST['luser'].lower()
        pass1=request.POST['lpass'].lower()
        user = auth.authenticate(username=uname, password=pass1)
        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            messages.info(request, "Invalid Credentials")
            return render(request, "login.html")
    return render(request,'login.html')

def register(request):
    if request.method=="POST":
        username = (request.POST['user']).lower()
        fname = request.POST['fname'].lower()
        lname = request.POST['lname'].lower()
        email = request.POST['email']
        passwd = request.POST['pass']
        con_passwd = request.POST['confirm_pass']
            
        if passwd==con_passwd:
            if User.objects.filter(username=username).exists():
                messages.info(request, "Username already exists!")
                return render(request, "register.html")
            elif User.objects.filter(email=email).exists():
                messages.info(request, "Email already exists!")
                return render(request, "register.html")
            else:
                user = User.objects.create_user(username=username, first_name=fname,
                last_name=lname, email=email, password=passwd)
                user.save()
                return redirect('login')
        else:
            messages.info(request,"Password's do not match!")
            return render(request,"register.html")
    else:
        return render(request, "register.html")
    return render(request,'register.html')


def logout(request):
    auth.logout(request)
    return redirect('/')

    
def fertilizer(request):
    if request.method=="POST":
        Temparature=int(request.POST['temp'])
        Humidity=int(request.POST['humid'])
        Moisture=int(request.POST['moisture'])
        Soil_Type=request.POST['soil']
        Crop_Type=request.POST['crop']
        Nitrogen=int(request.POST['nitro'])
        Potassium=int(request.POST['potassium'])
        Phosphorous=int(request.POST['phos'])
        from sklearn.preprocessing import LabelEncoder
        l=LabelEncoder()
        l.fit_transform([Soil_Type,Crop_Type])
        soil = l.fit_transform([Soil_Type])
        crop = l.fit_transform([Crop_Type])
        import pandas as pd
        df=pd.read_csv(r"static\Dataset\fertilizer.csv")
        import matplotlib.pyplot as plt
        import seaborn as sns
        print(df.head())
        print(df.describe())
        
        
        plt.figure(figsize=(16,10))
        sns.countplot(x='Crop_Type',data=df)
        def plot_conti(x):
            fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(15,5),tight_layout=True)
            axes[0].set_title('Histogram')
            sns.histplot(x,ax=axes[0])
            axes[1].set_title('Checking Outliers')
            sns.boxplot(x,ax=axes[1])
            axes[2].set_title('Relation with output variable')
            sns.boxplot(y = x,x = df['Fertilizer Name'])
        def plot_cato(x):
            fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5),tight_layout=True)
            axes[0].set_title('Count Plot')
            sns.countplot(x,ax=axes[0])
            axes[1].set_title('Relation with output variable')
            sns.countplot(x = x,hue = df['Fertilizer Name'], ax=axes[1])    

        plot_conti(df['Temparature'])
        plt.show()
        plot_conti(df['Humidity '])
        plt.show()
        plot_conti(df['Moisture'])
        plt.show()
        #relation of soil type with Temperature 
        plt.figure(figsize=(10,5))
        sns.boxplot(x=df['Soil_Type'],y=df['Temparature'])          
        plt.show()
        plt.figure(figsize=(15,6))
        sns.boxplot(x=df['Soil_Type'],y=df['Temparature'],hue=df['Fertilizer Name'])
        plt.show()
        from sklearn.preprocessing import LabelEncoder
        LE= LabelEncoder()

        df['Soil_Type'] = LE.fit_transform(df['Soil_Type'])
        Soil_Type = pd.DataFrame(zip(LE.classes_,LE.transform(LE.classes_)),columns=['Original','Encoded'])
        Soil_Type = Soil_Type.set_index('Original')
        print(Soil_Type)
        LE1 =  LabelEncoder()
        df['Crop_Type'] = LE1.fit_transform(df['Crop_Type'])

        #creating the DataFrame
        Crop_Type = pd.DataFrame(zip(LE1.classes_,LE1.transform(LE1.classes_)),columns=['Original','Encoded'])
        Crop_Type = Crop_Type.set_index('Original')
        print(Crop_Type)
        from sklearn.preprocessing import LabelEncoder
        '''encode_ferti = LabelEncoder()
        df['Fertilizer Name'] = encode_ferti.fit_transform(df['Fertilizer Name'])

        #creating the DataFrame
        Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
        Fertilizer = Fertilizer.set_index('Original')
        print(Fertilizer)'''
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(df.drop('Fertilizer Name',axis=1),df['Fertilizer Name'],test_size=0.2,random_state=1)
        from sklearn.ensemble import RandomForestClassifier
        rand = RandomForestClassifier(random_state = 42)
        rand.fit(x_train,y_train)
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
        pred_rand = rand.predict(x_test)
        pred_rand
        y_pred = rand.predict(x_test)
        acc = accuracy_score(y_test,y_pred)
        print(acc)
        train_pred = rand.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(train_acc)
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, classification_report

        params = {
            'n_estimators':[300,400,500],
            'max_depth':[5,10,15],
            'min_samples_split':[2,5,8]
        }
        grid_rand = GridSearchCV(rand,params,cv=3,verbose=3,n_jobs=-1)

        grid_rand.fit(x_train,y_train)

        pred_rand = grid_rand.predict(x_test)

        print(classification_report(y_test,pred_rand))

        print('Best score : ',grid_rand.best_score_)
        print('Best params : ',grid_rand.best_params_)
        from sklearn.ensemble import RandomForestClassifier
        rand = RandomForestClassifier(n_estimators=300,min_samples_split=2,max_depth=5,random_state = 42)
        rand.fit(x_train,y_train)
        y_pred = rand.predict(x_test)
        acc = accuracy_score(y_test,y_pred)
        print(acc)
        train_pred = rand.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(train_acc)
        import numpy as np
        pred_data=np.array([[Temparature,Humidity ,Moisture,soil,crop,Nitrogen,Potassium,Phosphorous]],dtype=object)
        prediction = rand.predict(pred_data)
        print(prediction)
        return render(request,"predict.html",{"Temparature":Temparature,"Humidity":Humidity,"Moisture":Moisture,"soil":Soil_Type,"crop":Crop_Type,"Nitrogen":Nitrogen,"Potassium":Potassium,"Phosphorous":Phosphorous,"prediction":prediction})
    else:
        return render(request,"fertilizer.html")
    
def predict(request):
    return render(request,"predict.html")

def crop(request):
    if request.method=="POST":
        nitro=int(request.POST['nitro'])
        phos=int(request.POST['phos'])
        pottas=int(request.POST['pottas'])
        temp=float(request.POST['temp'])
        humid=float(request.POST['humid'])
        ph=float(request.POST['ph'])
        rain=float(request.POST['rain'])
        #Phosphorous=int(request.POST['phos'])
        #from __future__ import print_function
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report
        from sklearn import metrics
        from sklearn import tree
        import warnings
        df = pd.read_csv('static/Dataset/Crop_recommendation.csv')
        print(df.head())
        print(df.tail())
        print(df.size)
        print(df.columns)
        print(df['label'].unique())
        print(df.dtypes)
        print(df['label'].value_counts())
        features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
        target = df['label']
        labels = df['label']
        acc = []
        model = []
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
        from sklearn.tree import DecisionTreeClassifier

        DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

        DecisionTree.fit(Xtrain,Ytrain)

        predicted_values = DecisionTree.predict(Xtest)
        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('Decision Tree')
        print("DecisionTrees's Accuracy is: ", x*100)

        print(classification_report(Ytest,predicted_values))
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(DecisionTree, features, target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        DT_pkl_filename = 'DecisionTree.pkl'
        # Open the file to save as pkl file
        DT_Model_pkl = open(DT_pkl_filename, 'wb')
        pickle.dump(DecisionTree, DT_Model_pkl)
        # Close the pickle instances
        DT_Model_pkl.close()
        from sklearn.naive_bayes import GaussianNB

        NaiveBayes = GaussianNB()

        NaiveBayes.fit(Xtrain,Ytrain)

        predicted_values = NaiveBayes.predict(Xtest)
        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('Naive Bayes')
        print("Naive Bayes's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        score = cross_val_score(NaiveBayes,features,target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        NB_pkl_filename = 'NBClassifier.pkl'
        # Open the file to save as pkl file
        NB_Model_pkl = open(NB_pkl_filename, 'wb')
        pickle.dump(NaiveBayes, NB_Model_pkl)
        # Close the pickle instances
        NB_Model_pkl.close()
        from sklearn.svm import SVC

        SVM = SVC(gamma='auto')

        SVM.fit(Xtrain,Ytrain)

        predicted_values = SVM.predict(Xtest)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('SVM')
        print("SVM's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        # Cross validation score (SVM)
        score = cross_val_score(SVM,features,target,cv=5)
        print(score)
        from sklearn.linear_model import LogisticRegression

        LogReg = LogisticRegression(random_state=2)

        LogReg.fit(Xtrain,Ytrain)

        predicted_values = LogReg.predict(Xtest)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('Logistic Regression')
        print("Logistic Regression's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        score = cross_val_score(LogReg,features,target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        LR_pkl_filename = 'LogisticRegression.pkl'
        # Open the file to save as pkl file
        LR_Model_pkl = open(DT_pkl_filename, 'wb')
        pickle.dump(LogReg, LR_Model_pkl)
        # Close the pickle instances
        LR_Model_pkl.close()
        from sklearn.ensemble import RandomForestClassifier

        RF = RandomForestClassifier(n_estimators=20, random_state=0)
        RF.fit(Xtrain,Ytrain)

        predicted_values = RF.predict(Xtest)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('RF')
        print("RF's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        # Cross validation score (Random Forest)
        score = cross_val_score(RF,features,target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        RF_pkl_filename = 'RandomForest.pkl'
        # Open the file to save as pkl file
        RF_Model_pkl = open(RF_pkl_filename, 'wb')
        pickle.dump(RF, RF_Model_pkl)
        RF_Model_pkl.close()
        # Close the pickle instances
        plt.figure(figsize=[10,5],dpi = 100)
        plt.title('Accuracy Comparison')
        plt.xlabel('Accuracy')
        plt.ylabel('Algorithm')
        sns.barplot(x = acc,y = model,palette='dark') 
        plt.show()
        accuracy_models = dict(zip(model, acc))
        for k, v in accuracy_models.items():
            print (k, '-->', v)
        data = np.array([[nitro,ph,phos,humid,temp,pottas,rain]],dtype=object)            
        prediction = RF.predict(data)
        print(prediction)
        return render(request,"crop_predict.html",{"nitro":nitro,"phos":phos,"ph":ph,"pottas":pottas,"humid":humid,"temp":temp,"rain":rain,
                                                   "pred":prediction})
    else:
        return render(request,"crop.html")
    
def crop_predict(request):
    return render(request,"crop_predict.html")

    return render(request,"crop.html")