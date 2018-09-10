from time import sleep

def printProgressBar(iteration, total, prefix='', suffix='', decimals = 1, length = 100, fill ='*'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration/float(total)))
    filledlength = int(length*iteration//total)
    bar = fill * filledlength + '-' * (length - filledlength)
    print('\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix),end='\r')
    #if iteration==total:
        #print()



items = list(range(0,57))
l = len(items)

printProgressBar(0,1, prefix= 'Progress:', suffix = 'Complete', length = 50)
for i, item in enumerate(items):
    sleep(0.1)
    printProgressBar(i+1,l,prefix = 'Progress:', suffix = 'Complete', length = 50)


