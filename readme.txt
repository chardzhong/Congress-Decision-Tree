Decision tree classification of congressman/woman party membership based on 2018 congressional voting data.

Program builds decision tree model using voting records on 42 congressional votes and implements information gain (entropy) as criteria for selecting attributes to split on.

Training and Testing data are subsets taken from congress_data.csv

User inputted min examples per leaf node used to prevent overfitting.

To run: python3 decision_tree.py train.csv party (min examples) test.csv

Performance:
Min examples	Accuracy
5		0.967741935483871

