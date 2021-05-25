"""
@ Filename:       AssociationAnalysis.py
@ Author:         Ryuk
@ Create Date:    2019-05-27   
@ Update Date:    2019-06-02
@ Description:    Implement AssociationAnalysis
"""

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    '''
    Function:  createSingletonSet
    Description: create set which only contain one elements
    Input:  data              dataType: ndarray    description:  data
    Output: singleton_set     dataType: frozenset  description:  invariable set which only contain one elements
    '''
    def createSingletonSet(self, data):
        singleton_set = []
        for record in data:
            for item in record:
                if [item] not in singleton_set:
                    singleton_set.append([item])
        singleton_set.sort()
        singleton_set = list(map(frozenset, singleton_set))       # generate a invariable set
        return singleton_set

    '''
    Function:  calculateSupportDegree
    Description: calculate the support degree for a given candidate set
    Input:  data              dataType: ndarray   description:  data
            candidate_set     dataType: list      description:  candidate set
    Output: support_degree    dataType: dict      description:  dictionary contains all set support_degree(item, support)
            frequent_items    dataType: list      description:  frequent items set
    '''
    def calculateSupportDegree(self, data, candidate_set):
        sample_sum = len(data)
        data = map(set, data)                   # transform data into set

        # calculate the frequency of each set in candidate_set appearing in data
        frequency = {}
        for record in data:
            for element in candidate_set:
                if element.issubset(record):  # elements in record
                    frequency[element] = frequency.get(element, 0) + 1

        # calculate the support degree for each set
        support_degree = {}
        frequent_items = []
        for key in frequency:
            support = frequency[key]/sample_sum
            if support >= self.min_support:
                frequent_items.insert(0, key)
            support_degree[key] = support
        return frequent_items, support_degree

    '''
     Function:  createCandidateSet
     Description: create candidate set
     Input:  frequent_items     dataType: list  description:  frequent items set
             k                  dataType: int   description:  the number of elements to be compared
     Output: candidate_set      dataType: list  description:  candidate set
     '''
    def createCandidateSet(self, frequent_items, k):
        candidate_set = []
        items_num = len(frequent_items)
        # merge the sets which have same front k-2 element
        for i in range(items_num):
            for j in range(i+1, items_num):
                L1 = list(frequent_items[i])[: k-2]
                L2 = list(frequent_items[j])[: k-2]
                if L1 == L2:
                    candidate_set.append(frequent_items[i] | frequent_items[j])
        return candidate_set

    '''
     Function:  findFrequentItem
     Description: find frequenct items
     Input:  data              dataType: ndarray   description:  data
     Output: support_degree    dataType: dict      description:  dictionary contains support_degree(item, support)
             frequent_items    dataType: list      description:  frequent items set
     '''
    def findFrequentItem(self, data):
        singleton_set = self.createSingletonSet(data)
        sub_frequent_items, sub_support_degree = self.calculateSupportDegree(data, singleton_set)
        frequent_items = [sub_frequent_items]
        support_degree = sub_support_degree
        k = 2

        while len(frequent_items[k-2]) > 0:
            candidate_set = self.createCandidateSet(frequent_items[k-2], k)
            sub_frequent_items, sub_support_degree = self.calculateSupportDegree(data, candidate_set)
            support_degree.update(sub_support_degree)
            if len(sub_frequent_items) == 0:
                break
            frequent_items.append(sub_frequent_items)
            k = k + 1
        return frequent_items, support_degree

    '''
     Function:  calculateConfidence
     Description: calculate confidence and generate rules
     Input:  frequent_item      dataType: set         description:  one record of frequent_set
             support_degree     dataType: dict        description:  support_degree 
             candidate_set      dataType: frozenset   description:  invariable set which only contain one elements
             rule_list          dataType: list        description:  invariable set which only contain one elements
     Output: confidence         dataType: dict        description:  confidence
             rule               dataType: list         description:  items whose confidence is larger than min_confidence
     '''
    def calculateConfidence(self, frequent_item, support_degree, candidate_set, rule_list):
        rule = []
        confidence = []
        for item in candidate_set:
            temp = support_degree[frequent_item]/support_degree[frequent_item - item]
            confidence.append(temp)
            if temp >= self.min_confidence:
                rule_list.append((frequent_item - item, item, temp))  #
                rule.append(item)
        return rule

    '''
     Function:  mergeFrequentItem
     Description: merge frequent item and generate rules
     Input:  frequent_item      dataType: set         description:  one record of frequent_set
             support_degree     dataType: dict        description:  support_degree 
             candidate_set      dataType: frozenset   description:  candidate set
             rule_list          dataType: list        description:  the generated rules
     '''
    def mergeFrequentItem(self, frequent_item, support_degree, candidate_set, rule_list):
        item_num = len(candidate_set[0])
        if len(frequent_item) > item_num + 1:
            candidate_set = self.createCandidateSet(candidate_set, item_num+1)
            rule = self.calculateConfidence(frequent_item, support_degree, candidate_set, rule_list)
            if len(rule) > 1:
                self.mergeFrequentItem(frequent_item, support_degree, rule, rule_list)

    '''
     Function:  generateRules
     Description: generate association rules
     Input:  frequent_set       dataType: set         description:  one record of frequent_set
             support_degree     dataType: dict        description:  support_degree 
     Output: rules              dataType: list        description:  the generated rules
     '''
    def generateRules(self, frequent_set, support_degree):
        rules = []
        for i in range(1, len(frequent_set)):              # generate rule from sets which contain more than two elements
            for frequent_item in frequent_set[i]:
                candidate_set = [frozenset([item]) for item in frequent_item]
                if i > 1:
                    self.mergeFrequentItem(frequent_item, support_degree, candidate_set, rules)
                else:
                    self.calculateConfidence(frequent_item, support_degree, candidate_set, rules)
        return rules

    '''
      Function:  train
      Description: train the model
      Input:  train_data       dataType: ndarray   description: items
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items   dataType: list      description: frequent items set
    '''
    def train(self, data, display=True):
        frequent_set, support_degree = self.findFrequentItem(data)
        rules = self.generateRules(frequent_set, support_degree)

        if display:
            print("Frequent Items:")
            for item in frequent_set:
                print(item)
            print("_______________________________________")
            print("Association Rules:")
            for rule in rules:
                print(rule)
        return frequent_set, rules

class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count              # support
        self.parent = parent
        self.next = None               # the same elements
        self.children = {}

    def display(self, ind=1):
        print(''*ind, self.item, '', self.count)
        for child in self.children.values():
            child.display(ind+1)

class FPgrowth:
    def __init__(self, min_support=3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    '''
    Function:  transfer2FrozenDataSet
    Description: transfer data to frozenset type
    Input:  data              dataType: ndarray     description: train_data
    Output: frozen_data       dataType: frozenset   description: train_data in frozenset type
    '''
    def transfer2FrozenDataSet(self, data):
        frozen_data = {}
        for elem in data:
            frozen_data[frozenset(elem)] = 1
        return frozen_data

    '''
      Function:  updataTree
      Description: updata FP tree
      Input:  data              dataType: ndarray     description: ordered frequent items
              FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
              count             dataType: count       description: the number of a record 
    '''
    def updataTree(self, data, FP_tree, header, count):
        frequent_item = data[0]
        if frequent_item in FP_tree.children:
            FP_tree.children[frequent_item].count += count
        else:
            FP_tree.children[frequent_item] = FPNode(frequent_item, count, FP_tree)
            if header[frequent_item][1] is None:
                header[frequent_item][1] = FP_tree.children[frequent_item]
            else:
                self.updateHeader(header[frequent_item][1], FP_tree.children[frequent_item]) # share the same path

        if len(data) > 1:
            self.updataTree(data[1::], FP_tree.children[frequent_item], header, count)  # recurrently update FP tree

    '''
      Function: updateHeader
      Description: update header, add tail_node to the current last node of frequent_item
      Input:  head_node           dataType: FPNode     description: first node in header
              tail_node           dataType: FPNode     description: node need to be added
    '''
    def updateHeader(self, head_node, tail_node):
        while head_node.next is not None:
            head_node = head_node.next
        head_node.next = tail_node

    '''
      Function:  createFPTree
      Description: create FP tree
      Input:  train_data        dataType: ndarray     description: features
      Output: FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
    '''
    def createFPTree(self, train_data):
        initial_header = {}
        # 1. the first scan, get singleton set
        for record in train_data:
            for item in record:
                initial_header[item] = initial_header.get(item, 0) + train_data[record]

        # get singleton set whose support is large than min_support. If there is no set meeting the condition,  return none
        header = {}
        for k in initial_header.keys():
            if initial_header[k] >= self.min_support:
                header[k] = initial_header[k]
        frequent_set = set(header.keys())
        if len(frequent_set) == 0:
            return None, None

        # enlarge the value, add a pointer
        for k in header:
            header[k] = [header[k], None]

        # 2. the second scan, create FP tree
        FP_tree = FPNode('root', 1, None)        # root node
        for record, count in train_data.items():
            frequent_item = {}
            for item in record:                # if item is a frequent set， add it
                if item in frequent_set:       # 2.1 filter infrequent_item
                    frequent_item[item] = header[item][0]

            if len(frequent_item) > 0:
                ordered_frequent_item = [val[0] for val in sorted(frequent_item.items(), key=lambda val:val[1], reverse=True)]  # 2.1 sort all the elements in descending order according to count
                self.updataTree(ordered_frequent_item, FP_tree, header, count) # 2.2 insert frequent_item in FP-Tree， share the path with the same prefix

        return FP_tree, header

    '''
      Function: ascendTree
      Description: ascend tree from leaf node to root node according to path
      Input:  node           dataType: FPNode     description: leaf node
      Output: prefix_path    dataType: list       description: prefix path
              
    '''
    def ascendTree(self, node):
        prefix_path = []
        while node.parent != None and node.parent.item != 'root':
            node = node.parent
            prefix_path.append(node.item)
        return prefix_path

    '''
    Function: getPrefixPath
    Description: get prefix path
    Input:  base          dataType: FPNode     description: pattern base
            header        dataType: dict       description: header
    Output: prefix_path   dataType: dict       description: prefix_path
    '''
    def getPrefixPath(self, base, header):
        prefix_path = {}
        start_node = header[base][1]
        prefixs = self.ascendTree(start_node)
        if len(prefixs) != 0:
            prefix_path[frozenset(prefixs)] = start_node.count

        while start_node.next is not None:
            start_node = start_node.next
            prefixs = self.ascendTree(start_node)
            if len(prefixs) != 0:
                prefix_path[frozenset(prefixs)] = start_node.count
        return prefix_path

    '''
    Function: findFrequentItem
    Description: find frequent item
    Input:  header               dataType: dict       description: header [name : (count, pointer)]
            prefix               dataType: dict       description: prefix path
            frequent_set         dataType: set        description: frequent set
    '''
    def findFrequentItem(self, header, prefix, frequent_set):
        # for each item in header, then iterate until there is only one element in conditional fptree
        header_items = [val[0] for val in sorted(header.items(), key=lambda val: val[1][0])]
        if len(header_items) == 0:
            return

        for base in header_items:
            new_prefix = prefix.copy()
            new_prefix.add(base)
            support = header[base][0]
            frequent_set[frozenset(new_prefix)] = support

            prefix_path = self.getPrefixPath(base, header)
            if len(prefix_path) != 0:
                conditonal_tree, conditional_header = self.createFPTree(prefix_path)
                if conditional_header is not None:
                    self.findFrequentItem(conditional_header, new_prefix, frequent_set)

    '''
     Function:  generateRules
     Description: generate association rules
     Input:  frequent_set       dataType: set         description:  current frequent item
             rule               dataType: dict        description:  an item in current frequent item
     '''
    def generateRules(self, frequent_set, rules):
        for frequent_item in frequent_set:
            if len(frequent_item) > 1:
                self.getRules(frequent_item, frequent_item, frequent_set, rules)

    '''
     Function:  removeItem
     Description: remove item
     Input:  current_item       dataType: set         description:  one record of frequent_set
             item               dataType: dict        description:  support_degree 
     '''
    def removeItem(self, current_item, item):
        tempSet = []
        for elem in current_item:
            if elem != item:
                tempSet.append(elem)
        tempFrozenSet = frozenset(tempSet)
        return tempFrozenSet

    '''
     Function:  getRules
     Description: get association rules
     Input:  frequent_set       dataType: set         description:  one record of frequent_set
             rule               dataType: dict        description:  support_degree 
     '''
    def getRules(self, frequent_item, current_item, frequent_set, rules):
        for item in current_item:
            subset = self.removeItem(current_item, item)
            confidence = frequent_set[frequent_item]/frequent_set[subset]
            if confidence >= self.min_confidence:
                flag = False
                for rule in rules:
                    if (rule[0] == subset) and (rule[1] == frequent_item - subset):
                        flag = True

                if flag == False:
                    rules.append((subset, frequent_item - subset, confidence))

                if (len(subset) >= 2):
                    self.getRules(frequent_item, subset, frequent_set, rules)

    '''
      Function:  train
      Description: train the model
      Input:  train_data       dataType: ndarray   description: items
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items   dataType: list      description: frequent items set
    '''
    def train(self, data, display=True):
        data = self.transfer2FrozenDataSet(data)
        FP_tree, header = self.createFPTree(data)
        #FP_tree.display()
        frequent_set = {}
        prefix_path = set([])
        self.findFrequentItem(header, prefix_path, frequent_set)
        rules = []
        self.generateRules(frequent_set, rules)

        if display:
            print("Frequent Items:")
            for item in frequent_set:
                print(item)
            print("_______________________________________")
            print("Association Rules:")
            for rule in rules:
                print(rule)
        return frequent_set, rules


class Eclat:
    def __init__(self,min_support=3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    '''
      Function:  invert
      Description: invert the data and filter the items smaller than min_support
      Input:  data            dataType: list   description: items
      Output: frequent_item   dataType: dict   description: invert data
    '''
    def invert(self, data):
        invert_data = {}
        frequent_item = []
        support = []
        for i in range(len(data)):
            for item in data[i]:
                if invert_data.get(item) is not None:
                    invert_data[item].append(i)
                else:
                    invert_data[item] = [i]

        for item in invert_data.keys():
            if len(invert_data[item]) >= self.min_support:
                frequent_item.append([item])
                support.append(invert_data[item])
        frequent_item = list(map(frozenset, frequent_item))
        return frequent_item, support

    '''
    Function:  getIntersection
    Description: get intersection
    Input:  frequent_set      dataType: dict  description:  frequent set
            support           dataType: list   description: support of data
    Output: frequent_item     dataType: dict  description:  frequent_item 
    '''
    def getIntersection(self, frequent_item, support):
        sub_frequent_item = []
        sub_support = []
        k = len(frequent_item[0]) + 1
        for i in range(len(frequent_item)):
            for j in range(i+1, len(frequent_item)):
                L1 = list(frequent_item[i])[:k-2]
                L2 = list(frequent_item[j])[:k-2]
                if L1 == L2:
                    flag = len(list(set(support[i]).intersection(set(support[j]))))
                    if flag >= self.min_support:
                        sub_frequent_item.append(frequent_item[i] | frequent_item[j])
                        sub_support.append(list(set(support[i]).intersection(set(support[j]))))
        return sub_frequent_item, sub_support

    '''
     Function: findFrequentItem
     Description: find frequent item
     Input: frequent_item   dataType: list   description: frequent item
            support         dataType: list   description: support of data
            frequent_set   dataType: list   description: frequent set
    '''
    def findFrequentItem(self, frequent_item, support, frequent_set,support_set):
        frequent_set.append(frequent_item)
        support_set.append(support)

        while len(frequent_item) >= 2:
            frequent_item, support = self.getIntersection(frequent_item, support)
            frequent_set.append(frequent_item)
            support_set.append(support)

    '''
        Function:  generateRules
        Description: generate association rules
        Input:  frequent_set       dataType: set         description:  current frequent item
                rule               dataType: dict        description:  an item in current frequent item
        '''
    def generateRules(self, frequent_set, rules):
        for frequent_item in frequent_set:
            if len(frequent_item) > 1:
                self.getRules(frequent_item, frequent_item, frequent_set, rules)

    '''
     Function:  removeItem
     Description: remove item
     Input:  current_item       dataType: set         description:  one record of frequent_set
             item               dataType: dict        description:  support_degree 
     '''
    def removeItem(self, current_item, item):
        tempSet = []
        for elem in current_item:
            if elem != item:
                tempSet.append(elem)
        tempFrozenSet = frozenset(tempSet)
        return tempFrozenSet

    '''
     Function:  getRules
     Description: get association rules
     Input:  frequent_set       dataType: set         description:  one record of frequent_set
             rule               dataType: dict        description:  support_degree 
     '''
    def getRules(self, frequent_item, current_item, frequent_set, rules):
        for item in current_item:
            subset = self.removeItem(current_item, item)
            confidence = frequent_set[frequent_item] / frequent_set[subset]
            if confidence >= self.min_confidence:
                flag = False
                for rule in rules:
                    if (rule[0] == subset) and (rule[1] == frequent_item - subset):
                        flag = True

                if flag == False:
                    rules.append((subset, frequent_item - subset, confidence))

                if len(subset) >= 2:
                    self.getRules(frequent_item, subset, frequent_set, rules)
    '''
      Function:  train
      Description: train the model
      Input:  train_data       dataType: ndarray   description: items
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items   dataType: list      description: frequent items set
    '''
    def train(self, data, display=True):
        # get the invert data
        frequent_item, support = self.invert(data)
        frequent_set = []
        support_set = []

        # get the frequent_set
        self.findFrequentItem(frequent_item, support,frequent_set, support_set)

        # transfer support list into frequency
        data = {}
        for i in range(len(frequent_set)):
            for j in range(len(frequent_set[i])):
                data[frequent_set[i][j]] = len(support_set[i][j])

        rules = []
        self.generateRules(data, rules)

        if display:
            print("Frequent Items:")
            for item in frequent_set:
                print(item)
            print("_______________________________________")
            print("Association Rules:")
            for rule in rules:
                print(rule)
        return frequent_set, rules
