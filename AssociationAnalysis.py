"""
@ Filename:       AssociationAnalysis.py
@ Author:         Danc1elion
@ Create Date:    2019-05-27   
@ Update Date:    2019-05-28
@ Description:    Implement AssociationAnalysis
"""

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
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
     Function:  mergeFrequentItem
     Description: merge frequent item and generate rules
     Input:  frequent_item      dataType: set         description:  one record of frequent_set
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
      Input:  train_data       dataType: ndarray   description: features
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items    dataType: list      description:  frequent items set
    '''
    def train(self, data, display=True):
        frequent_set, support_degree = self.findFrequentItem(data)
        rules = self.generateRules(frequent_set, support_degree)

        if display:
            for i in rules:
                print(i)
        return frequent_set, rules
