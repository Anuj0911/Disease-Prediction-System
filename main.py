import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
import GUI

# Load the dataset
data = pd.read_csv('training.csv')
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
        s = s + disease
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

def create_gui():
    root = tk.Tk()
    root.title("Disease Prediction")

    # Add other GUI components here...
    tk.Label(root, text="Symptom 1:").grid(row=0, column=0)
    GUI.p = tk.Entry(root)
    GUI.p.grid(row=0, column=1)

    tk.Label(root, text="Symptom 2:").grid(row=1, column=0)
    GUI.en = tk.Entry(root)
    GUI.en.grid(row=1, column=1)

    tk.Label(root, text="Symptom 3:").grid(row=2, column=0)
    GUI.bb = tk.Entry(root)
    GUI.bb.grid(row=2, column=1)

    tk.Label(root, text="Symptom 4:").grid(row=3, column=0)
    GUI.ee = tk.Entry(root)
    GUI.ee.grid(row=3, column=1)

    tk.Label(root, text="Symptom 5:").grid(row=4, column=0)
    GUI.hh = tk.Entry(root)
    GUI.hh.grid(row=4, column=1)

    tk.Label(root, text="Prediction:").grid(row=6, column=0)
    GUI.final_result = tk.Entry(root, width=50)
    GUI.final_result.grid(row=6, column=1)

    tk.Label(root, text="Remedy:").grid(row=7, column=0)
    GUI.remedy_result = tk.Entry(root, width=50)
    GUI.remedy_result.grid(row=7, column=1)

    tk.Button(root, text="Predict", command=prediction).grid(row=8, column=1)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
