import pandas as pd
from preprocess import Preprocessor


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

