import numpy as np
import pandas as pd


class Preprocessor:

    def main(self):
        path = "../data/nba2k-full.csv"
        df: pd.DataFrame = self.clean_data(path)
        pd.set_option('display.max_columns', None)
        print(df.head())

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


if __name__ == '__main__':
    Preprocessor().main()
