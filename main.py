# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def import_dataset(file, base):
    import json_parse as jsc
    project_config = jsc.create_nested_object_from_json('./config.json')
    import pandas as pd
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


def data_pre_processing(df):
    import pandas as pd

    country_code = import_dataset("country", "remote")
    country_code = country_code[['id', 'alpha2', 'alpha3', 'name']]
    country_code['alpha3'] = country_code['alpha3'].str.upper()
    df['name'] = 'NULL'
    for i in range(len(df)):
        if df.iloc[i]['country'] == 'ALL':
            continue
        else:
            if country_code.loc[
                country_code['alpha3'] ==
                    df.iloc[i, df.columns.get_loc('country')], 'name'].empty:
                name = 'NULL'
            else:
                name = country_code.loc[
                    country_code['alpha3'] == df.iloc[i, df.columns.get_loc('country')], 'name'].iloc[0]

            df.iloc[i, df.columns.get_loc('name')] = name

    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Transitional_reservations
    for i in df[(df['name'] == 'NULL') & (df['country'] == 'ANT')].index:
        df.iloc[i, df.columns.get_loc('name')] = "Netherlands Antilles"

    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Examples
    for i in df[(df['name'] == 'NULL') & (df['country'] == 'XXX')].index:
        df.iloc[i, df.columns.get_loc('name')] = "Unspecified Nationality"

    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Transitional_reservations
    for i in df[(df['name'] == 'NULL') & (df['country'] == 'ZAR')].index:
        df.iloc[i, df.columns.get_loc('name')] = "Zaire"

    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df.insert(1, 'year', df.pop('year'))
    df.insert(2, 'month', df.pop('month'))

    covid19 = import_dataset("covid19", "remote")

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

    df_merged = pd.merge(df, covid19_selected, on=['year', 'month'], how='left')

    df_merged['covid'] = df_merged['covid'].fillna(0).astype(int)

    dfc = df_merged.copy()

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dfc['covid'] = scaler.fit_transform(dfc[['covid']])

    return dfc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = import_dataset("main", "remote")
    #print(dataset)
    print(data_pre_processing(dataset))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
