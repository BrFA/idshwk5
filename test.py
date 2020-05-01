from sklearn.ensemble import RandomForestClassifier


class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label
        self.domain_len = len(_name)
        numbers = 0
        for c in _name:
            if c.isdigit():
                numbers += int(c)
        self.num_sum = numbers

    def returnData(self):
        return [self.domain_len, self.num_sum]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

trainlist = []
testlist = []

def initTrain(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            trainlist.append(Domain(name, label))
    return trainlist


def initTest(filename):
    featurelist = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            count = 0
            for c in line:
                if c.isdigit():
                    count += int(c)
            testData = [len(line), count]
            testlist.append(line)
            featurelist.append(testData)
    return testlist, featurelist


def output(content, filename):
    with open(filename, "w") as f:
        for i in content:
            f.write(i + "\n")


def main():
    trainSet = initTrain("train.txt")
    testDomainName, testFeatureSet = initTest("test.txt")
    featureMatrix = []
    labelList = []
    for item in trainSet:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    classification = clf.predict(testFeatureSet)
    result = []
    for i in range(len(classification)):
        if classification[i] == 1:
            result.append(testDomainName[i] + ",dga")
        else:
            result.append(testDomainName[i] + ",notdga")
    output(result, "result.txt")


if __name__ == '__main__':
    main()
