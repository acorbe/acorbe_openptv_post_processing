
import matplotlib.pyplot as plt
import traj_processing as tp
import traj_correlation as tc
import pandas as pd
import numpy as np
from scipy import interpolate
import glob
import matplotlib.animation as ani
import scipy.signal as signal
import os
import shutil

import cv2


import Data_classes as dtCl
import subprocess

import json
parameters_glob = json.load(open('parameters.glob'))

FPS = parameters_glob['FPS'] #FPS = 15.0
DEFAULT_SMOOTHING = parameters_glob['DEFAULT_SMOOTHING'] #0

PATH_PREFIX = '/home/acorbe/datas/'

def connect_to_pictures_gen_P_DB():
    directory_set = dtCl.openDirectorySet()

    fast_temp = parameters_glob['FAST_TMP'] + parameters_glob['KINECT_IMDB']
    for i,day_c in enumerate(directory_set):
        #extract_all_images_to given location
        print "unpacking day %s " % day_c
        loc_dir = fast_temp + str(i) + '/'
        subprocess.call("./forward_shm_extract_all_pict_to_dir %s %s" % (day_c,loc_dir), shell=True)



def connect_to_pictures_show_pict_list(plist
                                       , copy_to_path = None):

    fast_temp = parameters_glob['FAST_TMP'] + parameters_glob['KINECT_IMDB']
    print plist

    imgs = []
    for ids in plist:
        day = ids[0]
        time = ids[1]
        final_path = fast_temp + "%d/*_%d.png" % (day,time)       
            
        fli = glob.glob(final_path)
        if copy_to_path is not None:
            shutil.copy2(fli[0],copy_to_path)
        
        print fli
        #print final_path
        im = cv2.imread(fli[0])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.equalizeHist(im)
        
        imgs.append([im, fli])


    tot_pict = len(imgs)

    flag = True
    idx = 0

    ext_cnt = 0
    while ext_cnt < 200 and flag:
        cv2.imshow("track",imgs[idx][0])
        print imgs[idx][1]
        k = cv2.waitKey(0)

        print k

        if k ==  100 or k == 1048676:# ord('d'):
            idx += 1
            
        elif k == 97 or k == 1048673: #ord('a'):
            idx -=1

        elif k == 114 or k == 1048690:#ord('e'):
            flag = False        
          
        idx = idx % tot_pict
        ext_cnt += 1

        

    #fig = plt.figure()
    #ani.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=500)

def cut_off_history(h
                    , sym_truncation
                    , min_length = 10):
    if sym_truncation > 0 and len(h) > min_length:
        return h[sym_truncation:-sym_truncation-1]
    else:
        return h


def print_track(full_trajs_record
                , fps=FPS
                , save_data=False
                , saving_path=None
                , smoothing=DEFAULT_SMOOTHING
                , work_with_detection_info = None
                , sym_truncation = 0
                , lomb_scargle = True):

    print fps
    track = cut_off_history(full_trajs_record['traj'],sym_truncation)
    print track
    measure = cut_off_history(full_trajs_record['fmeasure'],sym_truncation)
    track_name = full_trajs_record.name

    day_c = track_name[0]

    print 'track name is', track_name, ', day code:', day_c
    
    if save_data:    
        new_figure = lambda : plt.figure(figsize=(20, 20), dpi=200)
    else:
        new_figure = lambda : plt.figure()


    new_figure()
        
    
        
    xx = tp.query_track(track,tp.get_x)

        
    yy = tp.query_track(track,tp.get_y)
    uu = tp.query_track(track,tp.get_u)
    vv = tp.query_track(track,tp.get_v)
    ax = tp.query_track(track,tp.get_ax)
    ay = tp.query_track(track,tp.get_ay)

    tt_s = tp.query_track(track,tp.get_t)

    t_start = tt_s[0]
    t_end = tt_s[-1]

    print 'track (start,end)= (%d,%d)' % (t_start,t_end)

    finer_grid = 100
    
    if work_with_detection_info is not None:
        detection_DB = work_with_detection_info        
        slice_of_interest = detection_DB.loc[ (day_c,t_start):(day_c,t_end) ]        
        times = slice_of_interest['shot_time']
        
        
        print times, len(times)
        
        tdiff = times.diff()[1:].values.astype('timedelta64[ms]')
        tdiff_fl = tdiff.astype('float')*1e-3
        
        #fps_h = 1./tdiff_fl
        fps_h = tdiff_fl
        
        tt = times - times.iloc[0]
        
        tt = tt.values.astype('timedelta64[ms]').astype('float')        
        tt *= 1e-3
        print tt, type(tt),len(tt),len(track)
        
        #print "len tt(DB)", len(tt), " len xx", len(xx), " len tt_s", len(tt_s)
        t_start = tt[0]
        t_end = tt[-1]
        fps = 1

        tt_sm = np.linspace(t_start,t_end,finer_grid)
        

        t_val,x_val,x_tck = tp.generate_spline_out_of_traj_RT(track,tp.get_x,tt,smoothing = smoothing)
        t_val,y_val,y_tck = tp.generate_spline_out_of_traj_RT(track,tp.get_y,tt,smoothing = smoothing)

    else:

        tt = tt_s
        tt = (tt - tt[0])/fps
        tt_sm = np.linspace(t_start,t_end,finer_grid)
        t_val,x_val,x_tck = tp.generate_spline_out_of_traj(track,tp.get_x,fps,smoothing = smoothing)
        t_val,y_val,y_tck = tp.generate_spline_out_of_traj(track,tp.get_y,fps,smoothing = smoothing)


        

    xx_sm = interpolate.splev(tt_sm/fps,x_tck)
    yy_sm = interpolate.splev(tt_sm/fps,y_tck)

    # print xx_sm

    uu_sm = interpolate.splev(tt_sm/fps,x_tck,der=1)
    vv_sm = interpolate.splev(tt_sm/fps,y_tck,der=1)

    ax_sm = interpolate.splev(tt_sm/fps,x_tck,der=2)
    ay_sm = interpolate.splev(tt_sm/fps,y_tck,der=2)

        
    
    
    tt_sm = (tt_sm - tt_sm[0])/fps

    

    print full_trajs_record.name
    if save_data:
        to_dump = np.asarray(zip(tt,xx,yy,uu,vv,ax,ay,measure))
        np.savetxt('track_%s.csv' % str(full_trajs_record.name)
                ,to_dump
                ,delimiter=','
                ,header="t, x, y, u, v, ax, ay, mu")

    #print zip(xx,uu)
    #print zip(yy,vv)
    

    a = np.sqrt(np.asarray(ax)**2 + np.asarray(ay)**2)
    a_sm = np.sqrt(np.asarray(ax_sm)**2 + np.asarray(ay_sm)**2)

    plt.subplot(2,2,1)
    plt.plot(xx,yy)
    plt.plot(xx_sm,yy_sm)
    
    
    plt.quiver(xx,yy,uu,vv,a,width=.005,alpha=.5)#,vmin=0, vmax=3)
    #plt.quiver(xx_sm,yy_sm,uu_sm,vv_sm,a_sm,width=.005,alpha=.5)#,vmin=0, vmax=3)

    plt.legend(['rl','sp'])
    plt.grid()
    plt.axis('equal')
    plt.colorbar()
    plt.title('track (color = |a|), t [%d,%d], units [m], [s]'%(t_start,t_end))

    
    
    plt.subplot(2,2,2)
    plt.plot(tt,xx)
    plt.plot(tt,yy)
    #plt.plot(tt_sm,xx_sm,alpha=.5)
    #plt.plot(tt_sm,yy_sm,alpha=.5)
    plt.grid()
    #plt.legend(['x','y','xs','ys'])
    plt.legend(['x','y'])#,'xs','ys'])
    
    

    out_u,out_t = tc.Correlations.local_structure_function(track)
    S2_tool = pd.DataFrame(zip(out_t,out_u),columns=['dt','S2'])
    S2 = S2_tool.groupby('dt').mean()

    slice_size = 2
    S2_f = S2[:slice_size]
    lcoeff = np.polyfit(np.log(S2_f.index/fps),np.log(S2_f),1)

    print lcoeff

    
    plt.subplot(4,2,6)
    plt.plot(S2.index/fps,S2)
    plt.plot(S2.index/fps,np.exp(np.polyval(lcoeff,np.log(S2.index/fps))))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend(['S2', 'm=%.2f'%(lcoeff[0]) ])

    
    plt.subplot(4,2,8)

    plt.plot(tt,measure)
    plt.grid()
    plt.legend(['#'])

    plt.subplot(2,2,3)
    plt.plot(tt[1:],fps_h)
    plt.grid()
    plt.legend('fps')

    if save_data and saving_path is not None:
        plt.savefig(saving_path + '/t_1.eps')


    
####
    
    #plt.figure()
    new_figure()

    plt.subplot(2,2,1)


    plt.plot(tt,uu)#,alpha=.5)
    plt.plot(tt,vv)#,alpha=.5)

    speed = np.sqrt( np.asarray(uu)**2 + np.asarray(vv)**2 )
    plt.plot(tt,speed)
    
    #plt.plot(tt_sm,uu_sm)
    #plt.plot(tt_sm,vv_sm)
    
    plt.grid()
    #plt.legend(['u','v','us','vs'])
    plt.legend(['u','v','speed']) #,'us','vs'])


    plt.subplot(2,2,3)
    
    plt.plot(tt,ax)
    plt.plot(tt,ay)
    plt.plot(tt,a)
    #plt.plot(tt_sm,a_sm)
    plt.grid()

    plt.legend(['ax','ay','a'])#,'a_sm'])
    
    
    if lomb_scargle == False or work_with_detection_info is None:

        plt.subplot(2,2,2)
        ff_u,p_u = signal.periodogram(uu,fs=15)
        ff_v,p_v = signal.periodogram(vv,fs=15)
        plt.plot(ff_u,p_u)
        plt.plot(ff_v,p_v)
        plt.xlabel('Hz')

        plt.legend(['psd(u)','psd(v)'])

        plt.grid()

        plt.subplot(2,2,4)
        ff_ax, p_ax = signal.periodogram(ax,fs=15)
        ff_ay, p_ay = signal.periodogram(ay,fs=15)
        ff_a, p_a = signal.periodogram(a,fs=15)

        plt.plot(ff_ax,p_ax)
        plt.plot(ff_ay,p_ay)
        plt.plot(ff_a,p_a)
        plt.xlabel('Hz')

        plt.legend(['psd(ax)','psd(ay)','psd(a)'])
        plt.grid()

    else:

        print "computing lombscargle..."
        
        ff = np.linspace(.01,8,20)

        N_rascle = float(len(tt))
        
        rascle_norm = lambda x : np.sqrt(4.*x/N_rascle)
        
        plt.subplot(2,2,2)
        periodg_u = signal.lombscargle(tt,uu,ff)
        periodg_v = signal.lombscargle(tt,vv,ff)
        periodg_speed = signal.lombscargle(tt,speed,ff)
        

        plt.plot(ff,rascle_norm(periodg_u))
        plt.plot(ff,rascle_norm(periodg_v))
        plt.plot(ff,rascle_norm(periodg_speed))
        plt.xlabel('Hz')

        plt.legend(['psd(u)','psd(v)','psd(speed)'])

        plt.grid()

        plt.subplot(2,2,4)
        periodg_ax = signal.lombscargle(tt,ax,ff)
        periodg_ay = signal.lombscargle(tt,ay,ff)
        periodg_a = signal.lombscargle(tt,a,ff)
        

        plt.plot(ff,rascle_norm(periodg_ax))
        plt.plot(ff,rascle_norm(periodg_ay))
        plt.plot(ff,rascle_norm(periodg_a))
        plt.xlabel('Hz')

        plt.legend(['psd(ax)','psd(ay)','psd(a)'])
        plt.grid()


    if save_data and saving_path is not None:
        plt.savefig(saving_path + '/t_2.eps')


    new_figure()
    plt.subplot(2,2,1)
    plt.hist(ax)
    plt.xlabel('ax')
    plt.subplot(2,2,2)
    plt.hist(ay)
    plt.xlabel('ay')
    plt.subplot(2,2,3)
    plt.hist(a)
    plt.xlabel('a')

    if save_data and saving_path is not None:
        plt.savefig(saving_path + '/t_3.eps')
    
        
        
    #if save_data:
    #    plt.savefig('track_%s.eps'%str(full_trajs_record.name) )


    return tdiff
    

def info_on_track(crStats,tid
                , save_data=False
                , show_pict=False
                , work_with_detection_info=None
                , smoothing=DEFAULT_SMOOTHING
                , lomb_scargle = True
                , **kw):

    
        


    track = crStats.trajs.loc[tid]

    if save_data:
        cur_path = PATH_PREFIX + 'tr_' + str(track.name)
        try:
            os.makedirs(cur_path)
        except :
            print 'dir is already there!'
        
    else:
        cur_path = None
    
    print_track(track
                , save_data=save_data
                , saving_path=cur_path
                , smoothing=smoothing
                , work_with_detection_info = work_with_detection_info
                , lomb_scargle = lomb_scargle
                , **kw)

    if show_pict:
        plist =  [ [tid[0], p.t] for p in track['traj']   ]
        print plist
        
        if save_data:
            frame_path = cur_path + '/frames'
            try:
                os.makedirs(frame_path)
            except :
                print 'dir is already there!'

            connect_to_pictures_show_pict_list(plist,copy_to_path = frame_path)
        else:
            connect_to_pictures_show_pict_list(plist)


if __name__ == '__main__':
    
    tid = (0,1200)
    track = crStats.trajs['traj'][tid]

    print_track(track)


