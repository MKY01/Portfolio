# DSTA 2018-19
#
# Solution for the discretional
# 'minipandas' project
#
# adapted by MKY
#

import numpy as np
import pandas as pd
import traceback
import sys

DATAFILE1 = "/Users/mankityip/Documents/BBK/Data-Science_Techniques_and_Applications/Week9/week9_lab/uk-districts-distances.csv"

DATAFILE2 = "/Users/mankityip/Documents/BBK/Data-Science_Techniques_and_Applications/Week9/week9_lab/2016-referendum-results.csv"

# UK electoral districts
N=382



def get_data(filename):
    data = pd.read_csv(filename)
    return data

def get_dist_matrix(data_dist):
    #order of areas should reflect the order of distances

    input_dists=[]

    areas=[]

    try:
        for row in range(N-1):
            #fetching 382 rows each time and advancing
            dists=list(data_dist.loc[row*N:(N*(row+1)-1)]['DIST_KM'])
            input_dists.append(dists)
            areas.append(data_dist.loc[row*N]['AREA_B'])

    except Exception as e:
        print(e)

    #inputing the last area:
    areas.append(data_dist['AREA_B'].iloc[-1])

    return input_dists, areas


def get_lambda_matrix(data_vote, areas, input_dists):
    '''Accesses the  leave  percentage and
        creates a vote data dictionary
    '''

    votes=dict()
    for index, row in data_vote.iterrows():
        votes[row['Area_Code']]=int(row['Leave'])
        #print(votes)

    #create a lambda matrix
    lam=list()

    #iterate distance matrix
    for i,row in enumerate(input_dists):
        v=list()
        for j,col in enumerate(row):
            try:
                if abs(votes[areas[i]]-votes[areas[j]])>0:
                    #formula for lambda
                    v.append(abs(votes[areas[i]]-votes[areas[j]])/(input_dists[i][j])**2)
                else:
                    #in case distance is 0
                    v.append(0)
            except:
                traceback.print_exc(file=sys.stdout)
                print(i,j)
                print(len(row))
                #sys.exit()

        lam.append(v)
    #print(lam[0])

    return lam

def find_argmax(areas,lambda_matrix):
    '''
	Iterate the lambda_matrix row by row.
	Find the maximum value for a row and populate maxima, maxima_areas.
    '''

	#Dictionary to hold the maximum lambda for a district
    maxima=dict()
	#Dictionary to hold the district name that has maximum lambda
    maxima_areas=dict()

    for r, row in enumerate(lambda_matrix):
        #get maximum of a row
        m=max(row)
		#find the index of the maximum
        idx=max(enumerate(row),key=lambda x: x[1])[0]
        #populate maxima areas with the index
        maxima_areas[areas[r]]=areas[idx]
        #populate maxima values
        maxima[areas[r]]=m

    print(maxima_areas)
    print(maxima)

    # EXERCISE:
    #
    #insert code to find the max of maxima:



if __name__ == '__main__':

    dist_data=get_data(DATAFILE1)
    vote_data=get_data(DATAFILE2)
    input_dists,areas=get_dist_matrix(dist_data)
    #our lambda matrix is aligned with distance matrix and areas list
    lambda_matrix=get_lambda_matrix(vote_data,areas,input_dists)
    find_argmax(areas,lambda_matrix)
