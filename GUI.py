import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
import re

# Load the dataset
data = pd.read_csv('Database/Training.csv')
df = pd.DataFrame(data)

cols = df.columns[:-1]
ll = list(cols)
pp = [str(i) for i in ll]

x = df[cols]  # x is the feature
y = df['prognosis']  # y is the target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

features = cols
feature_dict = {f: i for i, f in enumerate(features)}

latest_features = list(features).copy()
trix = {str(item).title().replace('_', ' '): item for item in latest_features}

remedies = {
    'Fungal infection': 'Keep the area clean and dry, apply antifungal creams.',
    'Allergy': 'Avoid allergens, take antihistamines.',
    'GERD': 'Eat smaller meals, avoid spicy foods, take antacids.',
    'Cholestasis': 'Consult a doctor, avoid alcohol, take prescribed medication.',
    'Drug Reaction': 'Stop taking the drug, take antihistamines, consult a doctor.',
    'Peptic ulcer': 'Avoid spicy foods, take antacids, consult a doctor.',
    'AIDS': 'Take antiretroviral drugs, maintain a healthy diet, consult a doctor.',
    'Diabetes': 'Monitor blood sugar levels, take insulin, maintain a healthy diet.',
    'Gastroenteritis': 'Stay hydrated, avoid solid foods for a while, take prescribed medication.',
    'Bronchial Asthma': 'Use inhalers, avoid allergens, consult a doctor.',
    'Hypertension': 'Reduce salt intake, exercise regularly, take prescribed medication.',
    'Migraine': 'Rest in a dark, quiet room, take pain relievers, consult a doctor.',
    'Cervical spondylosis': 'Exercise regularly, apply heat or cold, take pain relievers.',
    'Paralysis (brain hemorrhage)': 'Seek immediate medical attention, rehabilitation therapy.',
    'Jaundice': 'Stay hydrated, avoid alcohol, consult a doctor.',
    'Malaria': 'Take antimalarial drugs, avoid mosquito bites.',
    'Chicken pox': 'Use calamine lotion, avoid scratching, take antiviral medication.',
    'Dengue': 'Stay hydrated, take pain relievers, consult a doctor.',
    'Typhoid': 'Take antibiotics, avoid contaminated food and water, rest.',
    'Hepatitis A': 'Rest, stay hydrated, avoid alcohol.',
    'Hepatitis B': 'Take antiviral medication, avoid alcohol, consult a doctor.',
    'Hepatitis C': 'Take antiviral medication, avoid alcohol, consult a doctor.',
    'Hepatitis D': 'Take antiviral medication, avoid alcohol, consult a doctor.',
    'Hepatitis E': 'Rest, stay hydrated, avoid alcohol.',
    'Alcoholic hepatitis': 'Stop drinking alcohol, take prescribed medication, consult a doctor.',
    'Tuberculosis': 'Take antibiotics for several months, rest, consult a doctor.',
    'Common Cold': 'Rest, stay hydrated, take over-the-counter cold remedies.',
    'Pneumonia': 'Take antibiotics, rest, stay hydrated.',
    'Hemorrhoids(piles)': 'Use over-the-counter creams, take warm baths, eat a high-fiber diet.',
    'Heart attack': 'Seek immediate medical attention, take prescribed medication.',
    'Varicose veins': 'Wear compression stockings, elevate your legs, exercise regularly.',
    'Hypothyroidism': 'Take thyroid hormone replacement, consult a doctor.',
    'Hypoglycemia': 'Eat or drink fast-acting carbohydrates, monitor blood sugar levels.',
    'Arthritis': 'Take pain relievers, exercise regularly, consult a doctor.',
    'Paroxysmal Positional Vertigo': 'Perform Epley maneuver, avoid sudden head movements.',
    'Acne': 'Keep your skin clean, use over-the-counter acne treatments.',
    'Urinary tract infection': 'Take antibiotics, drink plenty of water.',
    'Psoriasis': 'Use topical treatments, avoid triggers, consult a doctor.',
    'Impetigo': 'Keep the area clean, use antibiotic ointment, avoid close contact with others.',
    'Osteoarthristis': 'Exercise regularly, take pain relievers, maintain a healthy weight.'
}

def prediction():
    symptoms = [p.get(), en.get(), bb.get(), ee.get(), hh.get()]
    symptoms = [trix[j] for j in symptoms if j != '']

    hack_set = set()
    pos = [feature_dict[symptom] for symptom in symptoms]

    sample_x = [1.0 if i in pos else 0.0 for i in range(len(features))]
    sample_x = [sample_x]

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    hack_set.add(*map(str, dt.predict(sample_x)))

    naive = GaussianNB()
    naive.fit(x_train, y_train)
    hack_set.add(*map(str, naive.predict(sample_x)))

    random = RandomForestClassifier()
    random.fit(x_train, y_train)
    hack_set.add(*map(str, random.predict(sample_x)))

    magic = list(hack_set)

    s = ""
    remedy = ""
    if len(hack_set) == 1:
        disease = "".join(magic[0]).strip()
        print("Disease:", disease)
        print("Hack Set:", hack_set)

        s = s + disease
        if disease == "Hypertension":
            remedy = "Reduce salt intake, exercise regularly, take prescribed medication."
            print("Remedy for Hypertension:", remedy)
        else:
            remedy = remedies.get(disease, "")
    else:
        disease1 = "".join(magic[0])
        disease2 = "".join(magic[1])
        s = s + disease1 + ' or ' + disease2
        remedy = remedies.get(disease1, "") + ' / ' + remedies.get(disease2, "")

    if not symptoms:
        final_result.delete(0, tk.END)
        final_result.insert(0, "Invalid! No Disease Found")
        remedy_result.delete('1.0', tk.END)
        remedy_result.insert(tk.END, "")
    elif len(set(symptoms)) != len(symptoms):
        final_result.delete(0, tk.END)
        final_result.insert(0, "Invalid! Try with unique Symptoms")
        remedy_result.delete('1.0', tk.END)
        remedy_result.insert(tk.END, "")
    else:
        final_result.delete(0, tk.END)
        final_result.insert(0, s)
        remedy_result.delete('1.0', tk.END)
        remedy_result.insert(tk.END, remedy)

class AutocompleteEntry(tk.Entry):
    def __init__(self, autocompleteList, *args, **kwargs):
        self.listboxLength = 0
        self.parent = args[0]

        # Custom matches function
        if 'matchesFunction' in kwargs:
            self.matchesFunction = kwargs['matchesFunction']
            del kwargs['matchesFunction']
        else:
            def matches(fieldValue, acListEntry):
                pattern = re.compile(
                    '.*' + re.escape(fieldValue) + '.*', re.IGNORECASE)
                return re.match(pattern, acListEntry)

            self.matchesFunction = matches

        # Custom return function
        if 'returnFunction' in kwargs:
            self.returnFunction = kwargs['returnFunction']
            del kwargs['returnFunction']
        else:
            def selectedValue(value):
                print(value)

            self.returnFunction = selectedValue

        tk.Entry.__init__(self, *args, **kwargs)
        self.focus()

        self.autocompleteList = autocompleteList

        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = tk.StringVar()

        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.moveUp)
        self.bind("<Down>", self.moveDown)
        self.bind("<Return>", self.selection)
        self.bind("<Escape>", self.deleteListbox)

        self.listboxUp = False

    def deleteListbox(self, event=None):
        if self.listboxUp:
            self.listbox.destroy()
            self.listboxUp = False

    def select(self, event=None):
        if self.listboxUp:
            index = self.listbox.curselection()[0]
            value = self.listbox.get(tk.ACTIVE)
            self.listbox.destroy()
            self.listboxUp = False
            self.delete(0, tk.END)
            self.insert(tk.END, value)
            self.returnFunction(value)

    def changed(self, name, index, mode):
        if self.var.get() == '':
            self.deleteListbox()
        else:
            words = self.comparison()
            if words:
                if not self.listboxUp:
                    self.listboxLength = len(words)
                    self.listbox = tk.Listbox(self.parent,
                                              width=self["width"], height=self.listboxLength)
                    self.listbox.bind("<Button-1>", self.selection)
                    self.listbox.bind("<Right>", self.selection)
                    self.listbox.place(
                        x=self.winfo_x(), y=self.winfo_y() + self.winfo_height())
                    self.listboxUp = True
                else:
                    self.listbox.delete(0, tk.END)

                for w in words:
                    self.listbox.insert(tk.END, w)
            else:
                self.deleteListbox()

    def selection(self, event):
        if self.listboxUp:
            self.var.set(self.listbox.get(tk.ACTIVE))
            self.listbox.destroy()
            self.listboxUp = False
            self.icursor(tk.END)

    def moveUp(self, event):
        if self.listboxUp:
            if self.listbox.curselection() == ():
                index = '0'
            else:
                index = self.listbox.curselection()[0]

            self.listbox.selection_clear(first=index)
            index = str(int(index) - 1)
            if int(index) == -1:
                index = str(self.listboxLength - 1)

            self.listbox.see(index)  # Scroll!
            self.listbox.selection_set(first=index)
            self.listbox.activate(index)

    def moveDown(self, event):
        if self.listboxUp:
            if self.listbox.curselection() == ():
                index = '-1'
            else:
                index = self.listbox.curselection()[0]

            if index != tk.END:
                self.listbox.selection_clear(first=index)
                if int(index) == self.listboxLength - 1:
                    index = "0"
                else:
                    index = str(int(index) + 1)

                self.listbox.see(index)  # Scroll!
                self.listbox.selection_set(first=index)
                self.listbox.activate(index)

    def comparison(self):
        return [w for w in self.autocompleteList if self.matchesFunction(self.var.get(), w)]

def matches(fieldValue, acListEntry):
    pattern = re.compile(re.escape(fieldValue) + '.*', re.IGNORECASE)
    return re.match(pattern, acListEntry)

root = tk.Tk()
root.geometry('1000x800')  # Adjusted window size
root.iconbitmap('Images/corona.ico')
root.title("Medical Disease Prediction")

autocompleteList = list(trix.keys())

frame = tk.LabelFrame(root, padx=10, pady=20, highlightthickness=2)
frame.pack(padx=20, pady=20)

c = tk.Label(frame, text="Medical Disease Prediction using ML", fg='blue4')
c.grid(row=0, column=0, columnspan=3, pady=(0, 20), padx=(30, 30), sticky="nsew")
c.config(font=("Consolas", 24, 'bold'))

L = tk.Label(frame, text='First Symptom:')
L.grid(row=1, column=0, sticky=tk.W, pady=(0, 10), padx=(30, 0))
L.config(font=("Consolas", 14))

p = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
p.grid(row=1, column=1)
p.config(font=('Consolas', 12, 'bold'))

def Destroy1():
    p.delete(0, tk.END)

one = tk.Label(frame, text='Second Symptom:')
one.grid(row=2, column=0, sticky=tk.W, pady=(0, 10), padx=(30, 0))
one.config(font=("Consolas", 14))
en = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
en.grid(row=2, column=1)
en.config(font=('Consolas', 12, 'bold'))

def Destroy2():
    en.delete(0, tk.END)

aa = tk.Label(frame, text='Third Symptom:')
aa.grid(row=3, column=0, sticky=tk.W, pady=(0, 10), padx=(30, 0))
aa.config(font=("Consolas", 14))

bb = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
bb.grid(row=3, column=1)
bb.config(font=('Consolas', 12, 'bold'))

def Destroy3():
    bb.delete(0, tk.END)

dd = tk.Label(frame, text='Fourth Symptom:')
dd.grid(row=4, column=0, sticky=tk.W, pady=(0, 10), padx=(30, 0))
dd.config(font=("Consolas", 14))

ee = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
ee.grid(row=4, column=1)
ee.config(font=('Consolas', 12, 'bold'))

def Destroy4():
    ee.delete(0, tk.END)

gg = tk.Label(frame, text='Fifth Symptom:')
gg.grid(row=5, column=0, sticky=tk.W, pady=(0, 10), padx=(30, 0))
gg.config(font=("Consolas", 14))

hh = AutocompleteEntry(
    autocompleteList, frame, width=32, matchesFunction=matches, fg='orange', bg='black', insertbackground='orange')
hh.grid(row=5, column=1)
hh.config(font=('Consolas', 12, 'bold'))

def Destroy5():
    hh.delete(0, tk.END)

def Destroy6():
    Destroy1()
    Destroy2()
    Destroy3()
    Destroy4()
    Destroy5()
    final_result.delete(0, tk.END)
    remedy_result.delete('1.0', tk.END)

kk = tk.Button(frame, text='Result', command=prediction, bg='red', fg='white', activebackground='red')
kk.config(font=('Consolas', '16', 'bold'))
kk.grid(row=6, column=1, pady=(10, 30), padx=(0, 80))


jj = tk.Button(frame, text=' Clear ', bg='red', fg='white', activebackground='red', command=Destroy6)
jj.config(font=('Consolas', '16', 'bold'))
jj.grid(row=6, column=1, padx=(50, 0), pady=(10, 30), columnspan=2)

final_result_label = tk.Label(frame, text='Disease:', fg='black')
final_result_label.grid(row=7, column=0, pady=(0, 10), padx=(30, 0))
final_result_label.config(font=("Consolas", 14))

final_result = tk.Entry(frame, width=50, borderwidth=0, bg='green', fg='white', justify=tk.CENTER, insertbackground='green')
final_result.grid(row=8, column=0, pady=(0, 20), padx=(30, 0), columnspan=2)
final_result.config(font=('Consolas', 14, 'bold'))
final_result.bind("<Key>", lambda e: "break")

remedy_result_label = tk.Label(frame, text='Remedies:', fg='black')
remedy_result_label.grid(row=9, column=0, pady=(0, 10), padx=(30, 0))
remedy_result_label.config(font=("Consolas", 14))

remedy_result = tk.Text(frame, width=50, height=4, borderwidth=0, bg='green', fg='white', wrap=tk.WORD)
remedy_result.grid(row=10, column=0, pady=(0, 20), padx=(30, 0), columnspan=2)
remedy_result.config(font=('Consolas', 14, 'bold'))
remedy_result.bind("<Key>", lambda e: "break")

tt = tk.Label(frame, text='N.B: Use 3 symptoms for better results')
tt.grid(row=11, column=0, columnspan=2, padx=(30, 0))
tt.config(font=('Consolas', 14, 'bold'))

root.mainloop()
