# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = import_dataset("main", "local")
    country_code = import_dataset("country", "local")
    covid19 = import_dataset("covid19", "local")

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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
