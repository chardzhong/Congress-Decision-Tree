import sys
import csv
import math

class DecisionNode:

    def __init__(self, test_name, test_index):
        self.test_name = test_name  # the name of the attribute to test at this node
        self.test_index = test_index  # the index of the attribute to test at this node

        self.children = {}  # dictionary mapping values of the test attribute to subtrees,
                            # where each subtree is either a DecisionNode or a LeafNode

    def classify(self, example):
        test_val = example[self.test_index]
        assert(test_val in self.children)
        return self.children[test_val].classify(example)

    def add_child(self, val, subtree):
        self.children[val] = subtree

    def to_str(self, level=0):
        prefix = "\t"*(level+1)
        s = prefix + "test: " + self.test_name + "\n"
        for val, subtree in sorted(self.children.items()):
            s += "{}\t{}={} ->\n".format(prefix, self.test_name, val)
            s += subtree.to_str(level+1)
        return s


class LeafNode:

    def __init__(self, pred_class, prob):
        self.pred_class = pred_class
        self.prob = prob

    def classify(self, example):
        return self.pred_class, self.prob

    def to_str(self, level):
        prefix = "\t"*(level+1)
        return "{}predicted class: {} ({})".format(prefix, self.pred_class, self.prob)


class DecisionTree:

    def __init__(self, csv_path):
        with open(csv_path, 'r') as infile:
            csvreader = csv.reader(infile)
            self.feature_names = next(csvreader)
            self.data = [row for row in csvreader]
            self.domains = [list(set(x)) for x in zip(*self.data)]
        self.root = None

    #trims data, in this case removes values not yay or nay and replaces with majority
    def trim(self, data, target_name):
        for attr in self.feature_names:
            if attr == target_name:
                continue
            for x in range(0, len(data)):
                col = self.feature_names.index(attr)
                if data[x][col] != "Yea" and data[x][col] != "Nay":
                    y = self.proportion(data, attr, "Yea")
                    n = self.proportion(data, attr, "Nay")
                    data[x][col] = "Yea" if y>=n else"Nay"
        self.domains = [list(set(x)) for x in zip(*self.data)]

    def learn(self, target_name, min_examples=0):
        self.trim(self.data, target_name)
        self.root = self.treehelper(target_name, min_examples, self.data, [])

    def treehelper(self, target_name, min_examples, subset, usedattr):
        bestattr = ""
        bestindex = -1
        bestgain = 0
        bestchilds = []
        for x in self.feature_names:
            if x == target_name or x in usedattr:
                continue
            childtuples = []
            col = self.feature_names.index(x)
            for val in self.domains[col]:
                childtuples.append([val, self.split(subset, col, val)])
            infogain = self.infogain(target_name, subset, childtuples)
            for child in childtuples:
                if len(child[1]) < min_examples:
                    infogain = -1
            if infogain > bestgain:
                bestattr = x
                bestindex = self.feature_names.index(x)
                bestgain = infogain
                bestchilds = childtuples
        #if cannot subset to > min examples, returns leaf node
        if bestindex == -1:
            predclass = ""
            prob =  0
            for x in self.domains[self.feature_names.index(target_name)]:
                prop = self.proportion(subset, target_name, x) 
                if prop > prob:
                    predclass = x
                    prob = prop
            return LeafNode(predclass, prob) 
        else:
            node = DecisionNode(bestattr, bestindex)
            usedattr.append(bestattr)
            for child in bestchilds:
                node.add_child(child[0], self.treehelper(target_name, min_examples, child[1], usedattr))
            return node
    #splits set into subset with all values matching val in col
    def split(self, subset, col, val):
        n = list(subset)
        return [x for x in n if x[col] == val]
    #calc infromation gain given parent and children
    #where children is a list of children sets
    def infogain(self, target_name, parentset, childtuples):
        childent = 0
        for child in childtuples:
            prop = len(child[1])/len(parentset)
            childent = childent + prop*self.entropy(child[1], target_name)
        return self.entropy(parentset, target_name) - childent
    #calc entropy value for a set
    def entropy(self, subset, target_name):
        entropy = 0
        col = self.feature_names.index(target_name)
        for x in self.domains[col]:
            prop = self.proportion(subset, target_name, x)
            if prop == 0:
                return -1
            entropy = entropy - prop * math.log(prop)
        return entropy

    #calc porportion of subset that has target_value for target_name
    def proportion(self, subset, target_name, target_value):
        if len(subset) == 0:
            return 0
        count = 0
        col = self.feature_names.index(target_name)
        for x in subset:
            if x[col] == target_value:
                count = count + 1
        return count/len(subset)

    def classify(self, example):
        return self.root.classify(example)

    def __str__(self):
        return self.root.to_str() if self.root else "<empty>"


#############################################

if __name__ == '__main__':

    path_to_csv = sys.argv[1]
    class_attr_name = sys.argv[2]
    min_examples = int(sys.argv[3])
    test = sys.argv[4]
    model = DecisionTree(path_to_csv)
    model.learn(class_attr_name, min_examples)
    # print(model)
    header = []
    data = []
    with open(test, 'r') as infile:
            csvreader = csv.reader(infile)
            header = next(csvreader)
            data = [row for row in csvreader]

    def prop(subset, target_name, target_value, header):
        if len(subset) == 0:
            return 0
        count = 0
        col = header.index(target_name)
        for x in subset:
            if x[col] == target_value:
                count = count + 1
        return count/len(subset)

    #trim
    for attr in header:
            if attr == class_attr_name:
                continue
            for x in range(0, len(data)):
                col = header.index(attr)
                if data[x][col] != "Yea" and data[x][col] != "Nay":
                    y = prop(data, attr, "Yea", header)
                    n = prop(data, attr, "Nay", header)
                    data[x][col] = "Yea" if y>=n else"Nay"

    right = 0
    for x in data:
         if model.classify(x)[0] == x[header.index(class_attr_name)]:
            right = right + 1
    print("Accuracy " + str(right/len(data)))
            
