#Yazid Suleimman

import matplotlib
from matplotlib.dates import strpdate2num
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time
from functools import reduce
from __future__ import division
from matplotlib import style
style.use("ggplot")

## Raise Numpy float errors
np.seterr(over='ignore')

## GLOBAL VARIABLES

# Sets the number of ticks to group together for the analysis
ticks = 10

# Sets the minimum similarity for analysis
minSim =  90



totalStart = time.time()


## Loads dbGetForexRatesCSV.csv file int date, rate, bid, and ask objects
##date, bid, ask = np.loadtxt('data/GBPUSD1d.txt', unpack=True,delimiter=',',converters={0:mdates.strpdate2num('%Y%m%d%H%M%S')})

def convert_date(date_bytes):
    return mdates.strpdate2num('%Y-%m-%dT%H:%M:%S.000Z')(date_bytes.decode('ascii'))

def convert_date2(date_bytes):
    return mdates.strpdate2num('%Y%m%d%H%M%S')(date_bytes.decode('ascii'))
    
## Commented out in preparation for the new forex_rates API code
date, rate, bid, ask = np.loadtxt('BIL616 Project\data\dbGetForexRatesCSV.csv', unpack=True, delimiter=',', converters={0: convert_date})

##plot initial graph of all the data
def initGraph():
    date,bid,ask = np.loadtxt('BIL616 Project\data\GBPUSD1d.txt', unpack=True, delimiter=',', converters={0: convert_date2})

    ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)
    ax1.plot(date,bid)
    ax1.plot(date,ask)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    #####
    plt.grid(True)
    for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    #######
    ax1_2 = ax1.twinx()
    
    #ax1_2.plot(date, (ask-bid))
    
    ax1_2.fill_between(date, 0, (ask-bid), facecolor='g',alpha=.3)
    
    #ax1_2.set_ylim(0, 3*ask.max())
    #######
    
    plt.subplots_adjust(bottom=.23)
    #plt.grid(True)
    
    plt.show()
    
##Creates an avgLine Array that for each bid and ask, gets the average.
avgLine = ((bid+ask)/2)

## Every pattern is stored in the pattern array
patternAr = []

## As we make the pattern, we store the outcome of pattern recognition here
performanceAr = []

patForRec = []

## Very simple difference based on the input values 
def percentChange(startPoint, currentPoint):
    try:
      x = ((float(currentPoint)-startPoint)/abs(startPoint))*100.00
      ## Error handling since Numpy bitches about NAN
      if x == 0.0:
            return 0.0000000001
      else:
            return x
    except:
            return 0.000001

##This function processes the data from the CSV file to calculate the percent change and and future outcome measures    
def patternStorage():
    ## Getting Current Unix Time
    patStartTime = time.time()

    ##Need to stop 30 short in order to still allow for outcomes to be calculated
    x = len(avgLine)-(30)

    ##We are skipping the first 30 because our pattern is 30 ticks long
    y = ticks+1
    print ('avgLine is ', len(avgLine), 'X is: ', x, 'Y is: ', y)
    while y < x:
        pattern =[]
        ## Grabbing the percent change for each tick for the last 30 minutes for each Y

        tickCounter = ticks
        while tickCounter > 0:
            pattern.append(percentChange(avgLine[y-ticks], avgLine[y-(tickCounter-1)]))
            #print (pattern)
            tickCounter -= 1

        patternAr.append(pattern)
        ## Splicing the avgLine array to get indexes between (y+20) and (y+30)
        outcomeRange = avgLine[y+20:y+30]

        ##grab current avgLine
        currentPoint= avgLine[y]

        ##In order to avoid a negative infinity percent change, we gotta do this....
        ##try:
        ##Get the sum of outcomeRanges which is (y+20) to (y+30) and divides that by the number of elements in the array
        avgOutcome = reduce(lambda x, y: x+y, outcomeRange) / len(outcomeRange)

        ##except Exception, e:
        ##print str(e)
        ##avgOutcome = 0
        
        ##Percent difference between the current point and the average outcome of the +20 - +30 points.
        futureOutcome = percentChange(currentPoint, avgOutcome)
        performanceAr.append(futureOutcome)

        y += 1
    patEndTime = time.time()
    print ('Pattern Storage Took:', patEndTime - patStartTime, 'seconds for', len(patternAr) ,'records')
    patternRecognition()

## currentPattern grabs the percent change between the 31st record against the prior 30.
def currentPattern():
    # First we populate an array that has the percent change between the first 30 objects of the avgLine array

    negativeTickCounter = (ticks * -1)
    while negativeTickCounter < 0:
        patForRec.append(percentChange(avgLine[(ticks * -1) - 1], avgLine[negativeTickCounter]))
        negativeTickCounter += 1
    
    print ('Pat:::', patForRec)

def patternRecognition():
    # Maximum similarity percentage found
    maxSim = 0
    # Total number of Similar patterns found
    totSim = 0


    print ('Pattern Length ', len(patForRec))
    if len(patForRec) != 0:
        for eachPattern in patternAr:

            simCounter = 0
            simTotal = 0
            while simCounter < ticks:
                simTotal += (100.00 - abs(percentChange(eachPattern[simCounter],patForRec[simCounter])))
                simCounter += 1

            howsim = (simTotal)/float(ticks)

            if(howsim > maxSim):
                maxSim = howsim

            ## This determines how accurate a pattern must be to be determined a good pattern. 70 = 70%
            if howsim > minSim:
                totSim = totSim + 1
                patdex = patternAr.index(eachPattern)


                print ('####################')
                print ('####################')
                print ('Base Pattern')
                print (patForRec)
                print ('====================')
                print ('Comparison Pattern')
                print (eachPattern)
                print ('--------------------')
                print ('Similarity:', howsim, '%')
                print ('====================')
                print ('Predicted outcomes', performanceAr[patdex])


                xpCounter = 0
                xp = []
                while xpCounter < (ticks):
                    xp.append(xpCounter+1)
                    xpCounter += 1


                fig = plt.figure()
                plt.plot(xp, patForRec)
                plt.plot(xp, eachPattern)
                plt.show()
    print ('Total number of similar patterns found:', totSim)
    print ('Maximum similarity percentage:', float(maxSim))


def confusion_matrix():
    conf_arr = patternAr

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')
    
    width, height = conf_arr.shape
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    #cb = fig.colorbar(res)
    #alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #plt.xticks(range(width), alphabet[:width])
    #plt.yticks(range(height), alphabet[:height])
    ##plt.savefig('confusion_matrix.png', format='png')
    plt.show()


def Main():
    print ('The graph of the data')
    initGraph()
    print ('Program set to use', ticks, 'ticks')
    print ('Minimum similarity set to:', minSim)
    patternStorage()
    currentPattern()
    patternRecognition()
    print ('Lenght of Pattern Array is: ', len(patternAr))
    print ('Lenght of Performance Array is: ', len(performanceAr))

Main()

totalTime = time.time() - totalStart
print ('Entire Processing time took:', totalTime, 'seconds')