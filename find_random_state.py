
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

target_counts = {0: 40, 1: 40, 2: 40}

for rs in range(10000):
    if rs % 1000 == 0:
        print(f"Checking {rs}...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1), 
        df['target'], 
        test_size=0.2, 
        random_state=rs
    )
    counts = y_train.value_counts().to_dict()
    if counts == target_counts:
        print(f"Found random_state: {rs}")
        print(counts)
        break
else:
    print("Not found in range")
