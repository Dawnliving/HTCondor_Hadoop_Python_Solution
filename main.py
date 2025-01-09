import json_parse as jsc
import pandas as pd


def import_dataset(file, base):
    project_config = jsc.create_nested_object_from_json('./config.json')

    df = pd.DataFrame()

    if file == "main":
        if base == "local":

            df = pd.read_csv(project_config.dataset.arrivals.local)

            return df
        elif base == "remote":
            # from aws hadoop cluster hive engine
            from sqlalchemy import create_engine

            # Establish Hive connection
            engine = create_engine(project_config.dataset.arrivals.hive)

            # SQL query to fetch data
            query = "SELECT * FROM allarrival"

            # Load data into a Pandas DataFrame
            df = pd.read_sql(query, engine)

            return df
    elif file == "country":
        if base == "local":

            df = pd.read_csv(project_config.dataset.country.local)

            return df
        elif base == "remote":

            df = pd.read_csv(project_config.dataset.country.hadoop)

            return df
    elif file == "covid19":
        if base == "local":

            df = pd.read_csv(project_config.dataset.covid19.local)

            return df
        elif base == "remote":

            df = pd.read_csv(project_config.dataset.covid19.hadoop)

            return df

    return df


def data_pre_processing(country_code, dataset, covid19):
    country_code = country_code[['id', 'alpha2', 'alpha3', 'name']]
    country_code['alpha3'] = country_code['alpha3'].str.upper()
    dataset['name'] = 'NULL'
    for i in range(len(dataset)):
        if dataset.iloc[i]['country'] == 'ALL':
            continue
        else:
            if country_code.loc[
                country_code['alpha3'] ==
                dataset.iloc[i, dataset.columns.get_loc('country')], 'name'].empty:
                name = 'NULL'
            else:
                name = country_code.loc[
                    country_code['alpha3'] == dataset.iloc[i, dataset.columns.get_loc('country')], 'name'].iloc[0]

            dataset.iloc[i, dataset.columns.get_loc('name')] = name

    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Transitional_reservations
    for i in dataset[(dataset['name'] == 'NULL') & (dataset['country'] == 'ANT')].index:
        dataset.iloc[i, dataset.columns.get_loc('name')] = "Netherlands Antilles"

    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Examples
    for i in dataset[(dataset['name'] == 'NULL') & (dataset['country'] == 'XXX')].index:
        dataset.iloc[i, dataset.columns.get_loc('name')] = "Unspecified Nationality"

    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Transitional_reservations
    for i in dataset[(dataset['name'] == 'NULL') & (dataset['country'] == 'ZAR')].index:
        dataset.iloc[i, dataset.columns.get_loc('name')] = "Zaire"

    dataset['date'] = pd.to_datetime(dataset['date'])

    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month

    dataset.insert(1, 'year', dataset.pop('year'))
    dataset.insert(2, 'month', dataset.pop('month'))

    covid19['date'] = pd.to_datetime(covid19['date'])

    covid19['year'] = covid19['date'].dt.year
    covid19['month'] = covid19['date'].dt.month

    covid19.insert(1, 'year', covid19.pop('year'))
    covid19.insert(2, 'month', covid19.pop('month'))

    covid19['year'] = pd.to_datetime(covid19['date']).dt.year
    covid19['month'] = pd.to_datetime(covid19['date']).dt.month

    covid = covid19.groupby(['year', 'month'], as_index=False)['World'].sum()

    covid.rename(columns={'World': 'covid'}, inplace=True)

    covid19_selected = covid[['year', 'month', 'covid']]

    df_merged = pd.merge(dataset, covid19_selected, on=['year', 'month'], how='left')

    df_merged['covid'] = df_merged['covid'].fillna(0).astype(int)

    dfc = df_merged.copy()

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dfc['covid'] = scaler.fit_transform(dfc[['covid']])

    return dfc


def top_10_arrivals(top_ten_arrivals):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#32CD32', '#90EE90', '#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000']
    custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', colors)

    values = top_ten_arrivals['arrivals_millions']
    norm = plt.Normalize(values.min(), values.max())
    colors = custom_cmap(norm(values))

    # Create vertical bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_ten_arrivals['name'], top_ten_arrivals['arrivals_millions'], color=colors)

    # Customize the plot
    plt.title('Top 10 arrivals in Malaysia')
    plt.xlabel('Country')
    plt.ylabel('Arrivals (Millions)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.6f}M',  # Display 6 decimal places
                 ha='center', va='bottom', fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig("./output/top_10_arrivals.jpg")


def top_10_male_arrivals(top_ten_male_arrivals):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colors = ['#ADD8E6', '#87CEEB', '#4169E1', '#1E90FF', '#0000FF', '#00008B']
    custom_cmap = LinearSegmentedColormap.from_list('custom_red', colors)

    values = top_ten_male_arrivals['arrivals_male_millions']
    norm = plt.Normalize(values.min(), values.max())
    colors = custom_cmap(norm(values))

    # Create vertical bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_ten_male_arrivals['name'], top_ten_male_arrivals['arrivals_male_millions'], color=colors)

    # Customize the plot
    plt.title('Top 10 male arrivals in Malaysia')
    plt.xlabel('Country')
    plt.ylabel('Male Arrivals (Millions)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.6f}M',  # Display 6 decimal places
                 ha='center', va='bottom', fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.savefig("./output/top_10_male_arrivals.jpg")


def top_10_female_arrivals(top_ten_female_arrivals):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colors = ['#FFB6C1', '#FF6B6B', '#FF4040', '#DC143C', '#B22222', '#8B0000']
    custom_cmap = LinearSegmentedColormap.from_list('custom_red', colors)

    values = top_ten_female_arrivals['arrivals_female_millions']
    norm = plt.Normalize(values.min(), values.max())
    colors = custom_cmap(norm(values))

    # Create vertical bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_ten_female_arrivals['name'], top_ten_female_arrivals['arrivals_female_millions'], color=colors)

    # Customize the plot
    plt.title('Top 10 female arrivals in Malaysia')
    plt.xlabel('Country')
    plt.ylabel('Male Arrivals (Millions)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.6f}M',  # Display 6 decimal places
                 ha='center', va='bottom', fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.savefig("./output/top_10_female_arrivals.jpg")


def annual_visitor_count(arrivals_by_year):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    arrivals_by_year.plot(kind='line', marker='o', color='blue')
    plt.title('Yearly Tourist Arrivals Trend')
    plt.xlabel('Year')
    plt.ylabel('Total Arrivals')
    plt.grid(True)
    plt.savefig("./output/annual_visitor_count.jpg")


def monthly_visitor_counts(arrivals_by_month):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    arrivals_by_month.plot(kind='bar', color='orange')
    plt.title('Monthly Tourist Arrivals Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Arrivals')
    plt.savefig("./output/monthly_visitor_counts.jpg")


def gender_ratio_analysis(gender_totals):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    gender_totals.plot(kind='pie', autopct='%1.1f%%', labels=['Male', 'Female'], startangle=90,
                       colors=['skyblue', 'pink'])
    plt.title('Gender Distribution of Tourist Arrivals')
    plt.ylabel('')
    plt.savefig("./output/gender_ratio_analysis.jpg")


def yearly_trend_gender(gender_by_year):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    gender_by_year.plot(kind='line', marker='o')
    plt.title('Yearly Trend of Male and Female Tourist Arrivals')
    plt.xlabel('Year')
    plt.ylabel('Number of Arrivals')
    plt.legend(['Male', 'Female'])
    plt.grid(True)
    plt.savefig("./output/yearly_trend_gender.jpg")


def model_first(dfc):
    # data processing
    model1_dfc = dfc[dfc['country'] != 'ALL']

    country_monthly = model1_dfc.pivot_table(index='date', columns='country', values='arrivals', aggfunc='sum').fillna(0)

    country_monthly_transposed = country_monthly.transpose()

    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # k-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(country_monthly_transposed)
    country_monthly_transposed['kmeans_cluster'] = kmeans_labels
    for cluster in range(3):
        print(f"Countries in K-Means Cluster {cluster}:")
        print(country_monthly_transposed[country_monthly_transposed['kmeans_cluster'] == cluster].index.tolist())

    # agglomerative cluster
    agglo = AgglomerativeClustering(n_clusters=3)
    agglo_labels = agglo.fit_predict(country_monthly_transposed.drop(columns='kmeans_cluster'))
    country_monthly_transposed['agglo_cluster'] = agglo_labels

    for cluster in range(3):
        print(f"Countries in Agglomerative Cluster {cluster}:")
        print(country_monthly_transposed[country_monthly_transposed['agglo_cluster'] == cluster].index.tolist())

    # PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(country_monthly_transposed.drop(columns=['kmeans_cluster', 'agglo_cluster']))

    print("Explained variance ratio by PCA components:", pca.explained_variance_ratio_)

    # visualization
    # K-Means
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-Means Clustering Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.grid()
    plt.savefig("./output/model1_k_means.jpg")

    # Agglomerative Clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=agglo_labels, cmap='plasma')
    plt.title('Agglomerative Clustering Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.grid()
    plt.savefig("./output/model1_agglomerative_cluster.jpg")

    # model result
    for cluster in range(3):
        print(f"Countries in K-Means Cluster {cluster}:")
        print(country_monthly_transposed[country_monthly_transposed['kmeans_cluster'] == cluster].index.tolist())

    for cluster in range(3):
        print(f"Countries in Agglomerative Cluster {cluster}:")
        print(country_monthly_transposed[country_monthly_transposed['agglo_cluster'] == cluster].index.tolist())

    # evaluation
    from sklearn.metrics import silhouette_score

    kmeans_silhouette = silhouette_score(country_monthly_transposed.drop(columns=['kmeans_cluster', 'agglo_cluster']),
                                         kmeans_labels)
    agglo_silhouette = silhouette_score(country_monthly_transposed.drop(columns=['kmeans_cluster', 'agglo_cluster']),
                                        agglo_labels)

    print(f"K-Means Silhouette Score: {kmeans_silhouette}")
    print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette}")


def model_second(dfc):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Function to add continent feature
    def add_continent_feature(df_clean):

        import pycountry_convert as pc

        def map_continent(country_name):
            try:
                country_alpha2 = pc.country_name_to_country_alpha2(country_name)
                continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
                continent_map = {
                    "AF": "Africa",
                    "AS": "Asia",
                    "EU": "Europe",
                    "NA": "North America",
                    "SA": "South America",
                    "OC": "Oceania"
                }
                return continent_map.get(continent_code, "Other")
            except:
                return "Other"

        # Apply the mapping function to the country column
        df_clean['continent'] = df_clean['country'].apply(map_continent)

        # One-hot encode the continent feature
        df_clean = pd.get_dummies(df_clean, columns=['continent'], prefix='continent')

        return df_clean

    # Function to calculate the 3-month average of COVID data
    def calculate_covid_average(df_clean):

        df_clean = df_clean.sort_values(by=['country', 'date'])
        df_clean['covid_avg_3_months'] = df_clean.groupby('country')['covid'].transform(
            lambda group: group.shift(1).rolling(window=3, min_periods=1).mean()
        ).fillna(0)
        return df_clean

    # Function to create features
    def create_features(df_clean):
        df_clean['male_proportion'] = df_clean['arrivals_male'] / (df_clean['arrivals'] + 1e-5)
        df_clean['cumulative_arrivals'] = df_clean.groupby(['country', 'year'])['arrivals'].cumsum()
        df_clean['arrival_growth_rate'] = df_clean.groupby('country')['arrivals'].pct_change()
        df_clean['arrival_growth_rate'] = df_clean['arrival_growth_rate'].fillna(0).clip(-1, 10)
        df_clean['arrival_growth_rate_lag1'] = df_clean.groupby('country')['arrival_growth_rate'].shift(1)
        df_clean['country_total_rank'] = df_clean.groupby('date')['arrivals'].rank(ascending=False, method='dense')
        df_clean['country_total_rank_lag1'] = df_clean.groupby('date')['arrivals'].rank().shift(1)
        df_clean['male_proportion_lag1'] = df_clean['arrivals_male'].shift(1) / (df_clean['arrivals'].shift(1) + 1e-5)
        df_clean['covid_lag1'] = df_clean.groupby('country')['covid'].shift(1)
        df_clean = calculate_covid_average(df_clean)
        df_clean['baseline_arrivals'] = df_clean['baseline_arrivals'].fillna(0)
        df_clean['arrivals'] = df_clean['arrivals'].fillna(0)
        df_clean['recovery_rate'] = df_clean['arrivals'] / (df_clean['baseline_arrivals'] + 1e-5) * 100
        return df_clean

    # Recovery class distribution
    def plot_recovery_class_distribution(df_clean):
        class_counts = df_clean['recovery_class'].value_counts()
        # class_counts.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'orange', 'green'])

        plt.figure(figsize=(8, 6))
        plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'orange', 'green'])
        plt.title('Recovery Class Distribution')
        plt.xlabel('Recovery Class')
        plt.ylabel('Count')
        plt.savefig("./output/model2_recovery_class_distribution.jpg")

    # Data preprocessing function
    def preprocess_data(dfc):
        df_clean = dfc[dfc.groupby('country')['arrivals'].transform('sum') > 0].copy()
        baseline = df_clean[df_clean['date'] == '2020-01-01'][['country', 'arrivals']]
        baseline = baseline.rename(columns={'arrivals': 'baseline_arrivals'})
        df_clean = pd.merge(df_clean, baseline, on='country', how='left')
        df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year

        # Add continent feature
        df_clean = add_continent_feature(df_clean)

        # Create additional features
        df_clean = create_features(df_clean)

        # Create season feature
        season_map = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
                      6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall', 12: 'winter'}
        df_clean['season'] = df_clean['month'].map(season_map)
        df_clean = pd.get_dummies(df_clean, columns=['season'], prefix='season')

        # Create target variable
        conditions = [
            df_clean['recovery_rate'] > 80,
            (df_clean['recovery_rate'] >= 20) & (df_clean['recovery_rate'] <= 80),
            df_clean['recovery_rate'] < 20
        ]
        labels = ['fast', 'slow', 'not_recovered']
        df_clean['recovery_class'] = np.select(conditions, labels, default='not_recovered')

        # **Call the recovery class distribution plotting function**
        plot_recovery_class_distribution(df_clean)

        # Select final features
        features = ['male_proportion_lag1', 'country_total_rank_lag1', 'covid_lag1',
                    'arrival_growth_rate_lag1', 'year', 'covid_avg_3_months'] + \
                   [col for col in df_clean.columns if col.startswith('season') or col.startswith('continent')]

        X = df_clean[features].fillna(0)
        y = df_clean['recovery_class']
        return X, y

    def train_and_evaluate_model(X, y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        clf = XGBClassifier(
            n_estimators=900,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        clf.fit(X_train, y_train)
        y_pred_encoded = clf.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        feature_importances = clf.feature_importances_
        feature_importance_report = {
            feature: importance for feature, importance in zip(X.columns, feature_importances)
        }
        return {
            'model': clf,
            'classification_report': classification_report(y_test_decoded, y_pred),
            'confusion_matrix': confusion_matrix(y_test_decoded, y_pred),
            'feature_importance_report': feature_importance_report
        }

    # Confusion matrix
    def plot_confusion_matrix(conf_matrix, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig("./output/model2_confusion_matrix.jpg")

    # Main program
    X, y = preprocess_data(dfc)  # Assuming dfc is your raw data
    results = train_and_evaluate_model(X, y)

    # Print feature importance report
    print("Feature Importance Report:")
    for feature, importance in sorted(results['feature_importance_report'].items(), key=lambda item: item[1],
                                      reverse=True):
        print(f"{feature}: {importance:.4f}")

    # Print classification report
    print("\nClassification Report:\n", results['classification_report'])

    # Confusion matrix visualization
    plot_confusion_matrix(results['confusion_matrix'], ['fast', 'slow', 'not_recovered'])


if __name__ == '__main__':
    # product
    dataset = import_dataset("main", "remote")
    country_code = import_dataset("country", "remote")
    covid19 = import_dataset("covid19", "remote")

    # local test
    # dataset = import_dataset("main", "local")
    # country_code = import_dataset("country", "local")
    # covid19 = import_dataset("covid19", "local")

    # dataset after data preprocessing
    dfc = data_pre_processing(country_code, dataset, covid19)

    # EDA
    grouped_arrivals = dfc[dfc['country'] != "ALL"]
    # top_10_arrivals
    top_ten_arrivals = pd.DataFrame(grouped_arrivals.groupby('name')['arrivals'].sum().nlargest(10)).reset_index()
    top_ten_arrivals['arrivals_millions'] = top_ten_arrivals['arrivals'] / 1000000
    top_10_arrivals(top_ten_arrivals)
    # top_10_male_arrivals
    top_ten_male_arrivals = pd.DataFrame(
        grouped_arrivals.groupby('name')['arrivals_male'].sum().nlargest(10)).reset_index()
    top_ten_male_arrivals['arrivals_male_millions'] = top_ten_male_arrivals['arrivals_male'] / 1000000
    top_10_male_arrivals(top_ten_male_arrivals)
    # top_ten_female_arrivals
    top_ten_female_arrivals = pd.DataFrame(
        grouped_arrivals.groupby('name')['arrivals_female'].sum().nlargest(10)).reset_index()
    top_ten_female_arrivals['arrivals_female_millions'] = top_ten_female_arrivals['arrivals_female'] / 1000000
    top_10_female_arrivals(top_ten_female_arrivals)
    # annual visitor count
    arrivals_by_year = dataset.groupby('year')['arrivals'].sum()
    annual_visitor_count(arrivals_by_year)
    # monthly visitor counts
    arrivals_by_month = dataset.groupby('month')['arrivals'].sum()
    monthly_visitor_counts(arrivals_by_month)
    # gender ratio analysis
    gender_totals = dataset[['arrivals_male', 'arrivals_female']].sum()
    gender_ratio_analysis(gender_totals)
    # yearly gender trend
    gender_by_year = dataset.groupby('year')[['arrivals_male', 'arrivals_female']].sum()
    yearly_trend_gender(gender_by_year)

    # Modeling
    # Cluster countries based on their time series trends in visitor numbers
    model_first(dfc)

    # Country-Level Recovery Classification with Multi-Dimensional Features
    # Main program
    model_second(dfc)
