import glob
import pandas as pd
import re
import datetime as DT
import tarfile
import numpy as np
import Data_classes as dtCl
import sys




def build_detection_DB_full(preciseTimeSearch=True):
    directory_set = dtCl.openDirectorySet()

    #end_data = pd.DataFrame()
    db_init = False

    for day_id,day_c in enumerate(directory_set):

        print day_c
        #    day_c = '13.09.17'

        day_gr = re.search('([0-9]{2}).([0-9]{2}).([0-9]{2})',day_c)
        day_gr_i = [ int(gr) for gr in day_gr.groups() ]

        y_ = day_gr_i[0] + 2000
        m_ = day_gr_i[1]
        d_ = day_gr_i[2]

        print "Y=%d M=%d D=%d" % (y_,m_,d_)

        build_path = lambda d : '../%s/output/M_*.tar.gz' % d

        flist_tar = glob.glob(build_path(day_c))
        #flist_tar.sort()

        flist = []
        fcontent = []
        fcontent_frame = []

        for tfile in flist_tar:
            print tfile
            tar_ = tarfile.open(tfile)
            flist.extend(tar_.getnames())
            for memb in tar_.getmembers():
                file__ = tar_.extractfile(memb)
                np_ = int(file__.readline())
                fcontent.append(np_)
                fr = dtCl.Frame()
                for i in range(np_):                
                    vals = file__.readline().split()
                    d = dtCl.Detection(x=int(vals[1])
                                       , y=int(vals[2])
                                       , z=float(vals[3])
                                       , A=float(vals[4])
                                       , B=float(vals[5])
                                       , C=float(vals[6])
                                       , cx=float(vals[7])
                                       , cy=float(vals[8]))
                    fr.append(d)
                fcontent_frame.append(fr)

            tar_.close()                
            print len(flist)

        flist_trim = [ st.split('/')[-1].split('.')[0] for st in flist ]

        regex_ = '[a-zA-Z]_kSnap-([0-9]{2})-([0-9]{2})-([0-9]{2})-([0-9]{3})_([0-9]*)'


        def name_to_DT(nm):
            re_groups = re.search(regex_,nm).groups()
            re_groups = [int(g) for g in re_groups]
            ret = DT.datetime(y_,m_,d_,re_groups[0],re_groups[1],re_groups[2],re_groups[3]*1000)
            ret = np.datetime64(ret)
            return ret,re_groups[4]

        flist_re = [ name_to_DT(nm) for nm in flist_trim ]
    
        datas = [l[0] for l in flist_re]

        
        ids = [(day_id,l[1]) for l in flist_re] #day id + frame id
        #day_ts = pd.TimeSeries(datas)


        
        

        day_data = pd.DataFrame(zip(datas
                        , fcontent
                        , fcontent_frame)
                    , index = pd.MultiIndex.from_tuples(ids
                        , names=['day_id','frame'])
                    , columns=['shot_time','#content', 'content'])

        day_data = day_data.sortlevel(0)


        if preciseTimeSearch:
            
                print "precise time search enabled..."
            #try :
                precise_fname = '../%s/output/timestamps.txt' % day_c
                print "target: %s" % precise_fname
                with open(precise_fname) as fi:
                    print "file %s has been opened..." % precise_fname
                    preciseDatas = np.loadtxt( fi , dtype='int64')
                    first_id = day_data.index[0][1]
                    corresponding_precise_data_idx = np.where(preciseDatas[:,0] == first_id )[0][0]
                    print "first processed index is %d" % corresponding_precise_data_idx
                    print "first processed instant is %s" % str( day_data.iloc[0]['shot_time'] )
                    preciseDatas = preciseDatas[corresponding_precise_data_idx:,:]

                    
                    print "precise data loaded..."
                    dts = preciseDatas[:,2] - preciseDatas[0,2]

                    dts_timedelta = [ np.timedelta64( microseconds = d * 1000 ) for d in dts ]

                    
                    print "time offset computed..."
                    initial_time_real = np.datetime64(day_data.iloc[0]['shot_time'])

                    assert day_data.iloc[0].name[1] == preciseDatas[0,0]
                    

                    for d,dt in zip(day_data['shot_time'],dts_timedelta):
                        #assert d.index[1] == pdt[0]
                        d = initial_time_real + dt

                    print "precise timestamp substituted..."

            # except:
            #     print " "
            #     print "ERROR!"
            #     print "in precise timestamp part"
            #     print " "


        

        if db_init:
            end_data = pd.concat([end_data, day_data ])
        else:
            end_data = day_data
            db_init = True


    return end_data




def save_detection_DB(detection_DB):
    detection_DB_FNAME = 'detection_DB.h5'
    detection_DB = detection_DB.sortlevel(0)
    storage = pd.HDFStore(detection_DB_FNAME)
    storage['detection_DB'] = detection_DB
    storage.close()



def attach_correct_time_stamp(detectionDB , crStats):

    print "filtering detections..."
    detectionDB_filtered = detectionDB.loc[crStats.datas.index]
    print "done!"
    
    print "attaching timestamps..."
    crStats.datas['time_stamps'] = detectionDB_filtered['shot_time']
    print "done!"

    print "assigning timestaps to trajs..."

    def fz(tr):
        day_c = tr.name[0]
        
        
        traj = tr['traj']
        start_t = traj[0].t
        end_t = traj[-1].t

        #print day_c, start_t, end_t

        time_stamps = crStats.datas['time_stamps'].loc[ (day_c,start_t) : (day_c,end_t) ]
        #print time_stamps.values
        return [time_stamps.values.astype('datetime64[ms]')]

    trajwise_timestamps = crStats.trajs.apply(fz,axis=1)['traj']

    crStats.trajs['time_stamps'] = trajwise_timestamps
    crStats.trajs['time_stamps_deltas'] = crStats.trajs['time_stamps'].apply( lambda f: (f - f[0]).astype('float')*1e-3 )

    print "done!"

if __name__ == '__main__':
    
    detection_DB = build_detection_DB_full()
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
    
    print "update detectionDB file? [y,n]"
    while True:
        choice = raw_input().lower()
            
        if choice in yes:
            response = True
            break
        elif choice in no:
            response = False
            break
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")
    

    if response:
        save_detection_DB(detection_DB)
