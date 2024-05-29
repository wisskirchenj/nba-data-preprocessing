import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:

    def main(self):
        path = "../data/nba2k-full.csv"
        df: pd.DataFrame = self.clean_data(path)
        df = self.feature_data(df)
        df = self.multicol_data(df)
        X, y = self.transform_data(df)
        answer = {
            'shape': [X.shape, y.shape],
            'features': list(X.columns),
        }
        print(answer)

    def clean_data(self, path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
        df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
        df['team'] = df['team'].fillna('No Team')
        df['height'] = df['height'].apply(lambda s: s.split()[2]).astype(float)
        df['weight'] = df['weight'].apply(lambda s: s.split()[3]).astype(float)
        df['salary'] = df['salary'].str.replace('$', '').astype(float)
        df['country'] = np.where(df['country'] == 'USA', 'USA', 'Not-USA')
        df['draft_round'] = df['draft_round'].str.replace('Undrafted', '0')
        return df

    def feature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['version'] = df['version'].str.replace('NBA2k', '20')
        df['version'] = pd.to_datetime(df['version'], format='%Y')
        df['age'] = (df['version'].dt.year - df['b_day'].dt.year).astype(int)
        df['experience'] = (df['version'].dt.year - df['draft_year'].dt.year).astype(int)
        df['bmi'] = (df['weight'] / df['height'] ** 2).astype(float)
        df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)
        df = df.loc[:, (df.nunique() <= 50) | df.columns.isin(['bmi', 'salary'])]
        return df

    def multicol_data(self, df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
        target = 'salary'
        corr = df.drop(columns=target).corr(numeric_only=True).unstack()
        corr_pairs = corr[abs(corr) > threshold].index
        corr_pairs = set([tuple(sorted(x)) for x in corr_pairs if x[0] != x[1]])
        for pair in corr_pairs:
            if pair[0] in df.columns and pair[1] in df.columns:
                df = df.drop(columns=df[[target, *pair]].corr()[target].idxmin())
        return df

    def transform_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        num_cols = df.drop(columns='salary').select_dtypes('number')
        numerical = pd.DataFrame(StandardScaler().fit_transform(num_cols), columns=num_cols.columns)
        encoder = OneHotEncoder(sparse_output=False)
        categorical = pd.DataFrame(encoder.fit_transform(df.select_dtypes('object')), columns=np.block(encoder.categories_))
        return pd.concat([numerical, categorical], axis=1), df['salary']


if __name__ == '__main__':
    Preprocessor().main()
