import numpy as np
import pandas as pd

np.random.seed(42)
rural_disease_probs = [0.15, 0.12, 0.1, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05] + [
    0.02
] * 10
urban_disease_probs = [0.02] * 10 + [
    0.15,
    0.12,
    0.1,
    0.09,
    0.08,
    0.07,
    0.07,
    0.06,
    0.05,
    0.05,
]
diseases = [
    "Respiratory_Infections",
    "Diarrheal_Diseases",
    "Malaria",
    "HIV_AIDS",
    "Parasitic_Infections",
    "Malnutrition",
    "Maternal_Mortality",
    "Neonatal_Conditions",
    "Hypertension",
    "Rheumatic_Heart_Disease",
    "Cardiovascular_Diseases",
    "Cancers",
    "Mental_Health_Disorders",
    "Obesity",
    "Respiratory_Diseases",
    "Allergies",
    "Autoimmune_Diseases",
    "Neurodegenerative_Diseases",
    "Osteoporosis",
    "STIs",
]


def generate_dataset(n, disease_probs, water_access, sanitation, nutrition):
    data = {
        "WaterAccess": np.random.binomial(1, water_access, size=n),
        "Sanitation": np.random.binomial(1, sanitation, size=n),
        "Nutrition": np.random.normal(nutrition, 0.1, size=n),
    }
    for disease, prob in zip(diseases, disease_probs):
        data[disease] = np.random.binomial(1, prob, size=n)
    return pd.DataFrame(data)


rural_dataset = generate_dataset(1000, rural_disease_probs, 0.6, 0.6, 0.4)
urban_dataset = generate_dataset(1000, urban_disease_probs, 1, 0.9, 0.6)
print("Rural Dataset:")
print(rural_dataset.head())
print("\nUrban Dataset:")
print(urban_dataset.head())
rural_dataset.to_csv("rural_health_dataset.csv", index=False)
urban_dataset.to_csv("urban_health_dataset.csv", index=False)
