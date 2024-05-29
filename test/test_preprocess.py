import numpy as np
import pandas as pd
from preprocess import Preprocessor
import pytest


class TestCleanData:

    #  Verify that CSV file is read correctly into a DataFrame
    def test_csv_file_read_correctly(self, mocker):
        # Mocking pd.read_csv to return a predefined DataFrame
        mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
            'b_day': ['12/31/90', '02/02/91'],
            'draft_year': ['1990', '1991'],
            'team': [None, 'Lakers'],
            'height': ['6-11 / 2.11', '6-5 / 1.96'],
            'weight': ['242 lbs. / 109.8 kg.', '220 lbs. / 99.8 kg.'],
            'salary': ['$1000000', '$2000000'],
            'country': ['USA', 'Canada'],
            'draft_round': ['1', 'Undrafted']
        }))
        preprocessor = Preprocessor()
        df = preprocessor.clean_data('dummy_path')
        assert not df.empty
        assert df['team'].iloc[0] == 'No Team'
        assert df['height'].dtype == float
        assert df['height'].iloc[0] == 2.11
        assert df['weight'].dtype == float
        assert df['weight'].iloc[1] == 99.8
        assert df['salary'].dtype == float
        assert df['country'].iloc[0] == 'USA'
        assert df['country'].iloc[1] == 'Not-USA'
        assert df['draft_round'].iloc[1] == '0'

    #  Removes columns with high multicollinearity correctly
    def test_removes_high_multicollinearity(self):
        preprocessor = Preprocessor()
        data = {
            'salary': [100, 200, 300],
            'feature1': [1, 2, 3],  # max collinear with salary (not to drop)
            'feature2': [1.1, 2.2, 3],  # High multicollinearity with feature1
            'feature3': [10, 20, 10]
        }
        df = pd.DataFrame(data)
        result_df = preprocessor.multicol_data(df)
        assert 'feature2' not in result_df.columns

    #  Retains columns that do not exhibit high multicollinearity
    def test_retains_non_multicollinear_columns(self):
        preprocessor = Preprocessor()
        data = {
            'salary': [100, 200, 300, 400, 500],
            'feature1': [1, 0, -1, -2, -3],
            'feature2': [2, 1.9, 2.2, 2, 2.3],  # No high multicollinearity with feature1
            'feature3': [1, 20, 30, 100, 400]
        }
        df = pd.DataFrame(data)
        result_df = preprocessor.multicol_data(df, threshold=0.9)
        assert len(result_df.columns) == 4

    #  DataFrame with only one column besides the target
    def test_single_column_besides_target(self):
        preprocessor = Preprocessor()
        data = {
            'salary': [100, 200, 300],
            'feature1': [1, 2, 3]
        }
        df = pd.DataFrame(data)
        result_df = preprocessor.multicol_data(df)
        assert result_df.equals(df)

    #  DataFrame with all columns having perfect multicollinearity
    def test_all_columns_perfect_multicollinearity(self):
        preprocessor = Preprocessor()
        data = {
            'salary': [100, 104, 104],
            'feature1': [1, 2, 3],
            'feature2': [1, 2, 3],  # Perfect multicollinearity with feature1
            'feature3': [1, 2, 3]   # Perfect multicollinearity with feature1
        }
        df = pd.DataFrame(data)
        result_df = preprocessor.multicol_data(df)
        assert len(result_df.columns) == 2  # Only salary and one feature should remain

    #  DataFrame with no numeric columns besides the target
    def test_no_numeric_columns_besides_target(self):
        preprocessor = Preprocessor()
        data = {
            'salary': [100, 200, 300],
            'feature1': ['A', 'B', 'C'],
            'feature2': ['X', 'Y', 'Z']
        }
        df = pd.DataFrame(data)
        result_df = preprocessor.multicol_data(df)
        assert result_df.equals(df)

    #  correctly scales numerical columns using StandardScaler
    def test_correctly_scales_numerical_columns(self):
        preprocessor = Preprocessor()
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'V'],
            'num1': [1, 2, 3, 4],
            'num2': [4, 5, 6, 7],
            'salary': [1000, 2000, 3000, 4000]
        })
        X, y = preprocessor.transform_data(df)
        assert np.allclose(X[['num1', 'num2']].mean(), 0), "Numerical columns are not scaled correctly"
        pytest.approx(df['salary'], y)

    #  correctly encodes categorical columns using OneHotEncoder
    def test_correctly_encodes_categorical_columns(self):
        preprocessor = Preprocessor()
        df = pd.DataFrame({
            'num1': [1, 2, 4],
            'cat1': ['A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X'],
            'salary': [1000, 2000, 3000]
        })
        X, y = preprocessor.transform_data(df)
        expected_columns = ['A', 'B', 'X', 'Y']
        assert all(col in X.columns for col in expected_columns), "Categorical columns are not encoded correctly"
