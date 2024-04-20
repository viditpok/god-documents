import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Function to apply k-means clustering to a dataset
def apply_kmeans(dataset_path, n_clusters=4, random_state=42):
    dataset = pd.read_csv(dataset_path)

    # Preprocessing
    le = LabelEncoder()
    for column in ["WaterAccess", "Sanitation", "Nutrition"]:
        dataset[column] = le.fit_transform(dataset[column])

    # Selecting features for clustering
    features = dataset[["WaterAccess", "Sanitation", "Nutrition"]]
    features = pd.concat([features, dataset.filter(regex="^Disease_", axis=1)], axis=1)

    # Standardizing the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # Adding cluster labels to the dataset for further analysis
    dataset["Cluster"] = cluster_labels

    return dataset


# Paths to datasets
rural_dataset_path = "data/rural_health_dataset.csv"
urban_dataset_path = "data/urban_health_dataset.csv"

# Apply k-means to each dataset
rural_dataset = apply_kmeans(rural_dataset_path)
urban_dataset = apply_kmeans(urban_dataset_path)

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


# Function to create combined heatmap
def create_combined_heatmap(rural_dataset, urban_dataset, diseases):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, dataset, title in zip(
        axes, [rural_dataset, urban_dataset], ["Greenfield", "Harborview"]
    ):
        sns.heatmap(
            dataset.groupby("Cluster")[diseases].mean(),
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            ax=ax,
        )
        ax.set_title(f"Heatmap of Disease Prevalence by Cluster in {title}")
        ax.set_xlabel("Disease")
        ax.set_ylabel("Cluster")
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()


create_combined_heatmap(rural_dataset, urban_dataset, diseases)
