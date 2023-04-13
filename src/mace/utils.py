import os


'''
Makes the output directory - if nessecary.
Returns the path of that dir.
'''
def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path