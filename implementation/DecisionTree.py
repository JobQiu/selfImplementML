class DecisionTree(object):
    """docstring for DecisionTree."""

    def __init__(self):
        super(DecisionTree, self).__init__()


    def gini_index(self, groups, classes):
        """
        groups : for example [1,1,1,0,0,0] and [1,1,0,0,0,0]
        classes : for example 0 and 1

        """

        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

if __name__ == "__main__":
    d = DecisionTree()


    print(d.gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
    print(d.gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

    dataset = [[2.771244718,1.784783929,0],
    	[1.728571309,1.169761413,0],
    	[3.678319846,2.81281357,0],
    	[3.961043357,2.61995032,0],
    	[2.999208922,2.209014212,0],
    	[7.497545867,3.162953546,1],
    	[9.00220326,3.339047188,1],
    	[7.444542326,0.476683375,1],
    	[10.12493903,3.234550982,1],
    	[6.642287351,3.319983761,1]]
    split = d.get_split(dataset)

    print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
#%%
