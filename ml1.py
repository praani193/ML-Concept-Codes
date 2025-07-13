import math
from collections import Counter
def entropy(labels):
    label_counts = Counter(labels)
    total = len(labels)
    entropy_value = 0.0

    for count in label_counts.values():
        probability = count / total
        entropy_value -= probability * math.log2(probability)

    return entropy_value
def information_gain(data, split_attribute, target_attribute):
    total_entropy = entropy([row[target_attribute] for row in data])
    values = set(row[split_attribute] for row in data)

    weighted_entropy = 0.0
    for value in values:
        subset = [row for row in data if row[split_attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy([row[target_attribute] for row in subset])

    return total_entropy - weighted_entropy

# Function to build the decision tree using ID3 algorithm
def id3(data, attributes, target_attribute):
    labels = [row[target_attribute] for row in data]

    # If all labels are the same, return the label
    if len(set(labels)) == 1:
        return labels[0]

    # If no more attributes, return the most common label
    if not attributes:
        return Counter(labels).most_common(1)[0][0]

    # Find the attribute with the highest information gain
    gains = [(attribute, information_gain(data, attribute, target_attribute)) for attribute in attributes]
    best_attribute = max(gains, key=lambda x: x[1])[0]

    # Build the tree
    tree = {best_attribute: {}}
    remaining_attributes = [attribute for attribute in attributes if attribute != best_attribute]

    for value in set(row[best_attribute] for row in data):
        subset = [row for row in data if row[best_attribute] == value]
        subtree = id3(subset, remaining_attributes, target_attribute)
        tree[best_attribute][value] = subtree

    return tree

# Example Weather Dataset
weather_data = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'}
]
for d in weather_data:
    a =d.values()
    print(a)
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target_attribute = 'PlayTennis'
decision_tree = id3(weather_data, attributes, target_attribute)

import pprint
pprint.pprint(decision_tree)
