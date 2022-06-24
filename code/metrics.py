from bisect import bisect
import matplotlib.pyplot as plt

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

def view_loss(loss):
    plt.clf()
    plt.semilogy(loss, label="train_loss")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    return plt

def view_score(score):
    plt.clf()
    plt.semilogy(score, label="Validation score")
    plt.title('model score')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend()
    return plt