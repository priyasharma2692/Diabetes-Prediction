
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk




def prediction():
    a = int(e20.get())
    b = int(e21.get())
    c = int(e22.get())
    d = int(e23.get())
    e = int(e24.get())
    f = float(e25.get())
    g = float(e26.get())
    h = int(e27.get())
    input_data = (a, b, c, d, e, f, g, h)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    if prediction[0] == 0:
        label99 = Label(window, text="Non-Diabetic", font="times 30 bold")
        label99.place(x=350, y=350)
    else:
        label90 = Label(window, text="                                                ", font="times 30 bold")
        label90.place(x=350, y=350)
        label99 = Label(window, text="Diabetic", font="times 30 bold")
        label99.place(x=350, y=350)

# loading the diabetes dataset
diabetes_dataset=pd.read_csv("C:/Users/priya/OneDrive/Desktop/DiabetesPrediction/diabetes.csv")

#seperating the data and the labels
X=diabetes_dataset.drop(columns = 'Outcome',axis=1)
Y=diabetes_dataset['Outcome'] 

# data standardization

scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
X=standardized_data
#train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2,test_size=0.2,stratify=Y)
#training machine learning model
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#accuracy
X_testprediction=classifier.predict(X_test)
testing_data_accuracy=accuracy_score(X_testprediction,Y_test)
print('accuracy score of svm ',testing_data_accuracy)


window=Tk()
window.title("Diabetes Prediction By Priya")
window.geometry("1000x500")

my_notebook = ttk.Notebook(window)
my_notebook.pack()

label=Label(window,text="Make a Prediction",font="times 30 bold")
label.place(x=220,y=30)

label20=Label(window,text="Pregnancies",font="times 15 bold")
label20.place(x=25,y=150)
e20=Entry(window)
e20.place(x=25,y=175)

label21=Label(window,text="Glucose",font="times 15 bold")
label21.place(x=225,y=150)
e21=Entry(window)
e21.place(x=225,y=175)


label22=Label(window,text="Blood Pressure",font="times 15 bold")
label22.place(x=425,y=150)
e22=Entry(window)
e22.place(x=425,y=175)


label3=Label(window,text="Skin Thickness",font="times 15 bold")
label3.place(x=625,y=150)
e23=Entry(window)
e23.place(x=625,y=175)


label4=Label(window,text="Insulin",font="times 15 bold")
label4.place(x=25,y=200)
e24=Entry(window)
e24.place(x=25,y=225)


label5=Label(window,text="BMI",font="times 15 bold")
label5.place(x=225,y=200)
e25=Entry(window)
e25.place(x=225,y=225)


label6=Label(window,text="Diabetes Pedigree Function",font="times 15 bold")
label6.place(x=375,y=200)
e26=Entry(window)
e26.place(x=425,y=225)


label7=Label(window,text="Age",font="times 15 bold")
label7.place(x=625,y=200)
e27=Entry(window)
e27.place(x=625,y=225)


b22 = Button(window,text = "Make Prediction", width = 20, command = prediction)
b22.place(x=500,y=400)

window.mainloop()