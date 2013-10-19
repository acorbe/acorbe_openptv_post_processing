#!/usr/bin/python

""" read txt files from the crowd tracking experiment
	and writes them in _targets format
"""

import os
import re

import sys
import datetime as DT
import numpy as np


CROPPING_FRAME=(40,600,72,356)


#REMARK: at the beginning, there have been days in which we had problems in storage/network.
#in such cases, although the time in the file name is still correct, the counter can have been restarted.
#let's make sure that this is not the case.
#we will order the files according to date next.
def ensure_continuity_min_max(list_of_files):
    print "consistency checking...",
    regex_capture_fnumber = '^.*-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{3}_([0-9]*).txt'
    get_f_number = lambda fname: int(re.search(regex_capture_fnumber,fname).group(1))

    flag_numbering_inconsistence = False

    prv_number = get_f_number(list_of_files[0])

    min_list_f = prv_number
    max_list_f = prv_number

    for fname in list_of_files[1:]:
        #print fname
        cur_number=get_f_number(fname)
        #print fname, cur_number
        # if cur_number-prv_number != 1:
        #     print "WARNING: numbering inconsistence!"
        #     flag_numbering_inconsistence = True
        prv_number = cur_number
        min_list_f = min(cur_number,min_list_f)
        max_list_f = max(cur_number,max_list_f)

    print "done!"
    return (flag_numbering_inconsistence, min_list_f,max_list_f) 


class File_properties:
    def __init__(self\
                     ,fname = []\
                     ,idn = []
                     ,timings = [] ):
        self.fname = fname
        self.idn = idn
        self.timings = timings

def obtain_file_timing(list_of_files):
    print "obtaining time list"
    regex_capture_fnumber = '^.*-([0-9]{2})-([0-9]{2})-([0-9]{2})-([0-9]{3})_([0-9]*).txt'
    #get_f_number = lambda fname: int(re.search(regex_capture_fnumber,fname).group(1))
    
    full_list_info = [   ]
    for fname in list_of_files:
        re_group = re.search( regex_capture_fnumber , fname ).groups()
        nfile = File_properties(fname\
                                    , int(re_group[5])
                                    , DT.timedelta(hours=int(re_group[0])\
                                                       ,minutes=int(re_group[1])\
                                                       ,seconds=int(re_group[2])\
                                                       ,milliseconds = int(re_group[3]) \
                                                       )\
                                    )
        full_list_info.append(nfile) 

    return full_list_info

def echo_full_list_info(full_list_info,file_out,fileout_justtimings_np):
    pass

    
    
        
    


#dir structure is kept (processing files in python_scripts), everything is redirected first to ../../


if __name__ == '__main__':

    

    #data_basedir='../../'
    data_basedir = '../' + sys.argv[1] + '/'
    
    #tracking_basedir='../'
    tracking_basedir = data_basedir + '/output_trk/'

    addresses_fld=tracking_basedir+'addresses/'

    #path = './txt'
    path = data_basedir + './output_trk/extr'
    print "generating list of files"
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt') and filename.startswith('T_'): 
                list_of_files.append(os.sep.join([dirpath, filename]))
    print "done...", len(list_of_files) , 'files found!'
    

    (fcont,fmin,fmax) = ensure_continuity_min_max(list_of_files)
    #prompt_to_file = lambda fname,cont :  with print >>addresses_fld+fname, cont

    def prompt_to_file(fname,cont):
        with open(addresses_fld+fname,'w') as f:
            print >> f,cont

    prompt_to_file('fmin',fmin)
    prompt_to_file('fmax',fmax)


    FILE_PROC_REMINDER = 1000
    tot_files = len(list_of_files)


    for cur_file,f in enumerate(list_of_files):
        # f = txt[0]
        if cur_file % FILE_PROC_REMINDER == 0:
            print cur_file, tot_files

        n = f.split('_')[-1].split('.txt')[0]
        target = tracking_basedir + './img/cam1.'+n+'_targets'
        # rt_is = './res/rt_is.'+n

        with open(f) as tmp:
            """ read the data from the crowd tracking file
            according to the Content file description
            id X   Y    Z        A               B         C
            """
            nlines = int(tmp.readline())
            array = []
            for line in tmp:
                        content = [float(x) for x in line.split()]
                        x = content[1] # notice the exchange of x,y #FIXED!
                        y = content[2]
                        if x > CROPPING_FRAME[0] \
                          and x < CROPPING_FRAME[1]\
                           and y > CROPPING_FRAME[2]\
                            and y < CROPPING_FRAME[3]:
                            array.append([float(x) for x in line.split()])
                        else:
                            nlines-=1

        with open(target,"w+") as tmp:
            """ write target file
            fprintf(fid,'%4d %9.4f %9.4f %5d %5d %5d %5d %5d\n',[pnr',x,y,n,nx,ny,sumg,tnr]');
            """
            print >> tmp , "%d" % nlines
            for line in array:
                out = line[0:3] # id, x, y
                id = line[0]
                x = line[1] # notice the exchange of x,y#FIXED!!
                y = line[2] 
                out = [id, x, y]
                out += 5*[999]
                print >> tmp, "%d %9.4f %9.4f %5d %5d %5d %5d %5d" % (tuple(out))

