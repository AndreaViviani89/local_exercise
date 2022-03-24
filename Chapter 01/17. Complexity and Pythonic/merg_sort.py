# Python program for implementation of MergeSort


def merge(arr, A, p, r):
    n1 = p - A + 1
    n2 = r - p
 

    a = [0] * (n1)
    R = [0] * (n2)