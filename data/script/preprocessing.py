from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    columns = ['id', 'entity', 'sentiment', 'text']
    data = pd.read_csv('data/raw/twitter_validation.csv', header=None, names=columns)
    
    # data['sentiment'] = data['sentiment'].replace('Irrelevant', 'Neutral')
    print(data.head())

    output_path = Path('data/processed/twitter_validation.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
