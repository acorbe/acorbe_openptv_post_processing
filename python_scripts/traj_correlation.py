import pandas as pd
import numpy as np
from itertools import combinations
import traj_processing as tp
from scipy.ndimage.measurements import label
import scipy.ndimage.filters as filters

from itertools import groupby
import collections
import Data_classes as dtCl

import signal_processing as sp

import json

import matplotlib.pyplot as plt

parameters_glob = json.load(open('parameters.glob'))

DEFAULT_FNAME = 'ptv_fulldata.h5'
FPS = parameters_glob['FPS'] #15
DEFAULT_SLICE_SIZE = parameters_glob['DEFAULT_SLICE_SIZE_LINT'] #3

 
def sub_select(data_,slice_):
    if isinstance( slice_ , collections.MutableSequence ):    
        data_s = data_.loc[slice_]    
    else:
        #print "extrema based selection"
        data_s = data_[:slice_]
        
    return data_s
   


def produce_interpolant( data,data2 = None ,slice_size = DEFAULT_SLICE_SIZE ):
    
    data_s = sub_select(data,slice_size)

    if data2 is None:
        lcoeff = np.polyfit(np.log(data_s.index/FPS),np.log(data_s),1)

    else:
        data_s2 = sub_select(data2,slice_size) #data2[:slice_size]
        #print np.log(data_s2)
        #print np.log(data_s)
        lcoeff = np.polyfit(np.log(data_s.transpose().values[0]),np.log( data_s2.transpose().values[0]),1)

    return lcoeff
        
    

def plot_interpolant(data,col,data2=None,slice_size = DEFAULT_SLICE_SIZE):
    l_coeff = produce_interpolant(data,data2, slice_size = slice_size)
    if data2 is None:
        plt.plot(data.index/FPS,np.exp(np.polyval(l_coeff,np.log(data.index/FPS))),col,alpha=.5)
        data_s = sub_select(data,slice_size)
        plt.plot(data_s.index/FPS,data_s,col+ '+')
   
    else:
        plt.plot(data,np.exp(np.polyval(l_coeff,np.log(data))),col,alpha=.5)
        data_s = sub_select(data,slice_size) 
        data_s2 = sub_select(data2,slice_size)
        plt.plot(data_s,data_s2,col + '+')

    return l_coeff
        

class Criterium:
    def __init__(self,criterium,criterium_name=None):
        self.criterium = criterium
        if criterium_name is not None:
            self.criterium_name = criterium_name
        else:
            self.criterium_name = 'cr*'
        

def frames_to_traj(data,lastid=None,short_len_filter=0):
    """ converts data from frames of particles into 
    trajectories of particles, using unique identities
    """
    if lastid == None:
        # id = []
        # for f in data:
        #     for p in f:
        #         id.append(p.id)
        id_ = [p.id for f in data for p in f]
        
        lastid = max(id_)
                
    # prepare the dataset
    traj = [dtCl.Trajectory() for i in range(lastid+1)]    
    for frame in data:
        for particle in frame:
            try:
                traj[particle.id].append(particle)
            except:
                print len(traj)
                print particle
                print particle.id
                
                
    for t in traj:
        if len(t) < short_len_filter:
            traj.remove(t)


    return traj




class Correlations:        

    def __init__(self):
        self.datas = None
        self.trajs = None

        self.stats = {}
        self.sets = {}

        self.criteria = {}

        self.data_structures_initialized = False
        
        self.tot_acquired_days = 0
        self.day_ids = []

        #self.build_selection_criteria()


    def acquire_data_sets(self, data , traj = None , date=None):

        current_day = self.tot_acquired_days
        self.day_ids.append(current_day)
        
        self.tot_acquired_days += 1

        print 'current day:', current_day
        
        #building local_datas data structure        
        framewise_presence = [ len(f) for f in data ]
        framewise_timeindex = [ ( current_day, f[0].t )  for f in data ]

        local_index = pd.MultiIndex.from_tuples( framewise_timeindex
                                                 , names=['day_id','frame'] )
        local_datas = pd.DataFrame( zip(framewise_presence,data)
                                    , columns=['fmeasure','data']
                                    , index = local_index)

        print 'local data length', len(local_datas)

        if traj is None:
            print "Rebuilding traj for data backlinking..."
            traj = frames_to_traj(data)
        

        #building local_trajs data structure
        trajwise_length = [len(t) for t in traj]
        trajwise_presence = [[-1]  for t in traj]
        trajwise_id = [(current_day,t[0].id) for t in traj]

        local_indext = pd.MultiIndex.from_tuples( trajwise_id
                                                  , names=['day_id','p_id'])
        local_trajs = pd.DataFrame( zip(trajwise_length
                                        , traj
                                        , trajwise_presence)
                                    , columns=['length','traj','fmeasure']
                                    , index=local_indext)

        print 'local traj legth', len(local_trajs)        


        #merging into global structure
        if not self.data_structures_initialized:
            print 'initializing db...'
            self.datas = local_datas
            self.trajs = local_trajs
            self.data_structures_initialized = True
        else:
            print 'updating db...'
            #self.datas.append(local_datas)# = pd.concat([self.datas, local_datas])
            self.datas = pd.concat([self.datas, local_datas])
            #self.trajs.append(local_trajs)
            self.trajs = pd.concat([self.trajs, local_trajs])


        print 'current db total length - datas:', len(self.datas), ' trajs:', len(self.trajs)
        print ' '

    


    
        


    def deep_copy_datas(self):
        print "deep copying datas..."
        self.dc_datas = self.datas.copy(deep=True)
        print "done!"

        
    def restore_copy_datas(self):
        print "restoring datas deep copy..."
        self.datas = self.dc_datas.copy(deep=True)
        print "done!"
        

    def calculate_trajwise_stats(self):
        print "calculating trajwise stats..."
        
        self.trajs['u_median'] = \
          self.trajs['traj'].apply(lambda t: np.median([p.u for p in t]))

        self.trajs['abs_vel_mean'] = \
          self.trajs['traj'].apply(lambda t: np.mean(np.sqrt( [p.u**2 + p.v**2 for p in t] ) ))

        calc_acc_mod_hist = lambda t : np.sqrt( [p.ax**2 + p.ay**2 for p in t] )

        self.trajs['a_max'] = \
            self.trajs['traj'].apply(lambda t: np.nanmax(  np.sqrt( [p.ax**2 + p.ay**2 for p in t] )  ))
            
        self.trajs['a_rms'] = \
            self.trajs['traj'].apply(lambda t : np.sqrt( np.mean(  calc_acc_mod_hist(t)  )  )  )

        self.trajs['ax_max'] = \
            self.trajs['traj'].apply(lambda t : np.nanmax(np.abs( tp.generate_time_history(t,tp.get_ax)  )) )

        self.trajs['ay_max'] = \
            self.trajs['traj'].apply(lambda t : np.nanmax(np.abs( tp.generate_time_history(t,tp.get_ay)  )) )

        f_rms = lambda arr : np.sqrt(np.mean(np.asarray(arr)**2))

        self.trajs['ax_rms'] = \
            self.trajs['traj'].apply( lambda t : f_rms(tp.generate_time_history(t ,tp.get_ax)) )

        self.trajs['ay_rms'] = \
            self.trajs['traj'].apply( lambda t : f_rms(tp.generate_time_history(t ,tp.get_ay)) )


        self.datas['ax_max'] = \
            self.datas['data'].apply( lambda f : np.nanmax( [p.ax for p in f ]  ) )

        


        print "done!"


        print "joining trajwise stas with data"
        
        def fz(dt):
            #print dt
            day_c = dt.name[0]
            #print day_c            
            ret = [ self.trajs['u_median'].loc[ ( day_c , p.id ) ]  for p in dt['data'] ]
            return [ret]
                
        
        temp_u_median = self.datas.apply(fz,axis=1)
        self.datas['u_median'] = temp_u_median['data']

        #return temp_u_median        

        print "done!"

    @staticmethod
    def Nan_ify_track_idx(t,idx_set):
        for idx in idx_set:
            t[idx].u = np.nan
            t[idx].v = np.nan
            t[idx].ax = np.nan
            t[idx].ay = np.nan

    def Nan_ify_tracks_der_extrema(self,qty=3):
        def fz(t):
            qty_eff = min(qty,len(t)/2)
            # #print qty_eff
            # #print t 
            # for idx in range(-qty_eff,qty_eff):
            #     #print idx
            #     t[idx].u = np.nan
            #     t[idx].v = np.nan
            #     t[idx].ax = np.nan
            #     t[idx].ay = np.nan

            Correlations.Nan_ify_track_idx(t, range(-qty_eff,qty_eff))
        self.trajs['traj'].apply(fz)

    def Nan_ify_tracks_der_conditional(self,further_conditioning):
        def fz(t):
            Correlations.Nan_ify_track_idx(t, range( len(t)))
            
        self.trajs[further_conditioning]['traj'].apply(fz)


    #def subsample_Tracks(self, subsampling_step):
    #    def f(t):
    #        pass
    #    self.trajs['traj'].apply(f)
    
    
    def apply_gen_filter_on_tracks(self,filter_):
        print "applying filter on tracks..."
        read_f = [tp.get_x, tp.get_y]
        write_f = [tp.set_x, tp.set_y]

        rw_f_pairs = zip(read_f,write_f)

        def update_time_history(t):
            #print "called"
            for rf,wf in rw_f_pairs:
                x = tp.generate_time_history(t,rf)
                x_s = filter_(x) #filters.gaussian_filter1d(x,sigma)
                tp.overwrite_time_history(t,x_s,wf)


        self.trajs['traj'].apply(update_time_history)#,axis=1)
        print "done!"

        self.compute_derivatives_spline()

    def analyse_framerate(self , framerate_issue_thr = .1 ):
        self.trajs['dt_max'] = \
          self.trajs['time_stamps_deltas'].apply( lambda t : np.max(np.diff(t) ) )

        self.criteria['isolate_tracks_bad_framerate'] = \
          self.trajs['dt_max'] > framerate_issue_thr
            

    

    def apply_gaussian_filter_on_tracks(self,sigma):
        self.apply_gen_filter_on_tracks(filter_ = lambda x : filters.gaussian_filter1d(x,sigma))
   
    def apply_lowpass_filter_on_tracks(self,
                                        filter_taps = sp.generate_filter(FPS,1,6)):    
        self.apply_gen_filter_on_tracks(filter_ = lambda x : sp.filter_signal(x,filter_taps))
   

    # def apply_lowpass_filter_on_tracks(self,
    #                                     filter_taps = sp.generate_filter(FPS,1,6)):
    #     read_f = [tp.get_x, tp.get_y]
    #     write_f = [tp.set_x, tp.set_y]

    #     rw_f_pairs = zip(read_f,write_f)

    #     def update_time_history(t):
    #         #print "called"
    #         for rf,wf in rw_f_pairs:
    #             x = tp.generate_time_history(t,rf)
    #             x_s = sp.filter_signal(x,filter_taps) #filters.gaussian_filter1d(x,sigma)
    #             tp.overwrite_time_history(t,x_s,wf)


    #     self.trajs['traj'].apply(update_time_history)#,axis=1)    
        


    def compute_derivatives_spline(self,fps=FPS,smoothing=0,sub_sampling_step = 0):
        
        print "recalculating derivatives in terms of cubic splines..."
        tp.calculate_velocity_spline_based(self.trajs['traj'],fps,smoothing,sub_sampling_step)
        tp.calculate_acceleration_spline_based(self.trajs['traj'],fps,smoothing,sub_sampling_step)
        print "done!"

    def compute_derivatives_spline_RT(self,smoothing = 0,sub_sampling_step = 0):
        print "recalculating derivatives in terms of cubic splines RT..."
        tp.calculate_velocity_spline_based_RT(self.trajs['traj'],self.trajs['time_stamps_deltas'],smoothing,sub_sampling_step)
        tp.calculate_acceleration_spline_based_RT(self.trajs['traj'],self.trajs['time_stamps_deltas'],smoothing,sub_sampling_step)
        print "done!"


    def compute_derivatives_fd(self,fps=FPS,sub_sampling_step = 0):
        print "recalculating derivatives in terms of finite differencing"
        tp.calculate_velocity_fd_based(self.trajs['traj'],fps,sub_sampling_step)
        tp.calculate_acceleration_fd_based(self.trajs['traj'],fps,sub_sampling_step)
        print "done!"


    def compute_derivatives_fd_RT(self,sub_sampling_step = 0):
        print "recalculating derivatives in terms of finite differencing"
        tp.calculate_velocity_fd_based_RT(self.trajs['traj'],self.trajs['time_stamps_deltas'],sub_sampling_step)
        tp.calculate_acceleration_fd_based_RT(self.trajs['traj'],self.trajs['time_stamps_deltas'],sub_sampling_step)
        print "done!"

        


    def finalize_acquisition(self):
        print "finalizing acquisition (associating local frame measure to tracks)..."
        ## computing mutual presence along tracks
        #for idx,t in self.trajs.iterrows():
        #    print idx
        #    day = idx[0]
            #t['fmeasure'][:] = [self.datas.loc[(day,p.t),'fmeasure'] for p in  t['traj']]        
            #t['fmeasure'][:] = self.datas.loc[(day,t['traj'][])]
            #print frlist
            #t['fmeasure'] = frlist
            #print t['fmeasure']

        get_day = lambda t : t.name[0]
        get_traj = lambda t,idx : t['traj']
        
        def fz(t):
            #print (get_day(t),get_traj_t(t,0)), (get_day(t),get_traj_t(t,-1))

            #print self.datas.loc[
            #    (get_day(t),get_traj_t(t,0)) : (get_day(t),get_traj_t(t,-1))
            #    , 'fmeasure' ].values

            day = t.name[0]
            tr = t['traj']
            return [self.datas.loc[ \
                (day,tr[0].t) : (day,tr[-1].t)
                , 'fmeasure' ].values]

        
        #self.trajs.apply(fz,axis=1)
        measures = self.trajs.apply(fz,axis=1)['traj']
            
        #return measures

        self.trajs['fmeasure'] = measures
        #self.calculate_trajwise_stats()


        
        
        print "done!"

        
    def save(self):
        store = pd.HDFStore(DEFAULT_FNAME)
        
        store['datas'] = self.datas
        store['trajs'] = self.trajs
        #store['days_ids'] = self.day_ids
        #store[]
        store.close()


    def load(self):
        store = pd.HDFStore(DEFAULT_FNAME)
        self.datas = store['datas']
        self.trajs = store['trajs']

        self.data_structures_initialized = True        

        store.close()


        
    def build_selection_criteria(self):
        print "building selection criteria..."
        clambda_crit_isolate_tracks_ped_is_always_alone = \
          lambda fmeas : np.max(fmeas) == 1
        
        self.criteria['isolate_tracks_ped_is_always_alone'] =\
            self.trajs['fmeasure'].map(clambda_crit_isolate_tracks_ped_is_always_alone)

        self.crit_isolate_tracks_ped_is_always_alone = \
          self.trajs['fmeasure'].map(clambda_crit_isolate_tracks_ped_is_always_alone)
        
          
        clambda_crit_isolate_tracks_st_two_ped_happen_to_be_present = \
          lambda fmeas : np.max(fmeas) == 2
        self.crit_isolate_tracks_st_two_ped_happen_to_be_present = \
          self.trajs['fmeasure'].map(clambda_crit_isolate_tracks_st_two_ped_happen_to_be_present)

        self.criteria['isolate_tracks_st_two_ped_happen_to_be_present'] = \
            self.trajs['fmeasure'].map(clambda_crit_isolate_tracks_st_two_ped_happen_to_be_present)

        
        self.criteria['isolate_tracks_shorter_than_N'] = \
            lambda N: self.trajs['length'] < N

        self.crit_isolate_tracks_shorter_than_N = \
            self.criteria['isolate_tracks_shorter_than_N']


        self.criteria['isolate_tracks_longer_than_N'] = \
            lambda N: self.trajs['length'] > N


        self.crit_isolate_tracks_longer_than_N =\
            self.criteria['isolate_tracks_longer_than_N']

        X_LBOUND = -.9
        X_HBOUND = .9


        self.criteria['isolate_tracks_complete'] = \
          self.trajs['traj'].map(
              lambda t : (t[0].x <= X_LBOUND and t[-1].x >= X_HBOUND) 
                or (t[0].x >= X_HBOUND and t[-1].x <= X_LBOUND)
              )

        self.crit_isolate_complete_tracks = self.criteria['isolate_tracks_complete']


        self.criteria['isolate_frames_with_exactly_N_ped'] = \
            lambda N : self.datas['data'].map(lambda f : len(f) == N)

        self.criteria['isolate_frames_with_singles'] = \
            self.criteria['isolate_frames_with_exactly_N_ped'](1)

        self.criteria['isolate_frames_with_pairs'] = \
            self.criteria['isolate_frames_with_exactly_N_ped'](2)


        print "done!"

          


    @staticmethod
    def local_structure_function(track):
        #there is a good reason not to pass the FPS: the following groupby('dt') is then int-based rather than float based
        out_u = [] 
        out_t = []           
            
        for p1,p2 in combinations(track,2):
            #p1 = p[0]
            #p2 = p[1]
            delta_u = (p2.u - p1.u)**2 + (p2.v - p1.v)**2
            delta_t = p2.t - p1.t
            out_u.append(delta_u)
            out_t.append(delta_t)

        return out_u,out_t

    @staticmethod
    def local_structure_function_power(track,p):
        #there is a good reason not to pass the FPS: the following groupby('dt') is then int-based rather than float based
        out_u = [] 
        out_t = []           
            
        for p1,p2 in combinations(track,2):
            #p1 = p[0]
            #p2 = p[1]
            delta_u = ((p2.u - p1.u)**2 + (p2.v - p1.v)**2)**(p/2.)
            delta_t = p2.t - p1.t
            out_u.append(delta_u)
            out_t.append(delta_t)

        return out_u,out_t

        


    def single_ped_structure_function(self
                                      , fps = FPS
                                      , plot_out=False
                                      , structure_function = \
                                      lambda t : Correlations.local_structure_function(t)
                                      , further_conditioning = None
                                      , title_add = ''
                                      , color = 'b'
                                      , time_scale = 'time'
                                      ):
        
        self.build_selection_criteria()

        if time_scale == 'time':
            fps_n = fps
        elif time_scale == 'frame':
            fps_n = 1

        if further_conditioning is None:        
            self.tracks_with_always_singles = \
                self.trajs[self.crit_isolate_tracks_ped_is_always_alone]
        else:
            self.tracks_with_always_singles = \
              self.trajs[self.crit_isolate_tracks_ped_is_always_alone
                         & further_conditioning]


        self.stats['sin_mean_abs_vel'] = np.mean([ s
                     for t in self.tracks_with_always_singles['traj']
                     for s in tp.generate_velocity_speed_history(t) ])

        single_avg_vel = self.stats['sin_mean_abs_vel']
        


        global_single_stats = \
          self.tracks_with_always_singles['traj'].apply(structure_function)

                
        self.stats['single_ped_deltavel'] = []
        self.stats['single_ped_deltat'] = []

        final_t = self.stats['single_ped_deltat'] #[]#
        final_u = self.stats['single_ped_deltavel'] #[]

        for idx,item in global_single_stats.iteritems():
            final_u.extend(item[0])
            final_t.extend(item[1])


        self.stats['single_ped_S2'] = pd.DataFrame(zip(final_t,final_u)
                                                         ,columns=['dt','sq|dv|'])        


        a = self.stats['single_ped_S2'].groupby('dt').mean()
        

        if plot_out:
            plt.figure()
            plt.plot(a.index/fps_n,a/single_avg_vel,color)
            plt.xscale('log')
            plt.yscale('log')

            plt.title('S2 for single pedestrian'+title_add)
            
            slice_size = 2
            S2_f = a[:slice_size]/single_avg_vel
            lcoeff = np.polyfit(np.log(S2_f.index/fps_n),np.log(S2_f),1)

            plt.plot(a.index/fps_n,np.exp(np.polyval(lcoeff,np.log(a.index/fps_n))),color)
            plt.grid()

            plt.legend(['S2 single','m=%g'%lcoeff[0]])

            plt.savefig('singleS2'+title_add +'.eps')

            
            print lcoeff

            

            to_dump = np.asarray(zip(a.index.values/fps_n,a.values/single_avg_vel))

            
            np.savetxt('singleS2'+title_add +'.csv'
                       , to_dump
                       , delimiter=','
                       , header="dt, <(|dv|/<v>)^p>")

            

        return a


    def single_ped_compare_S2S4S6(self
                                  ,further_conditioning = None):        


        tracks_subfilters = further_conditioning
        mean_abs_sin_vel = self.stats['sin_mean_abs_vel']

        

        S2 = self.single_ped_structure_function(further_conditioning = tracks_subfilters)

        S4 = self.single_ped_structure_function(
            structure_function
            = lambda track : self.local_structure_function_power(track,4.)
            , further_conditioning =tracks_subfilters)



        S6 = self.single_ped_structure_function(
            structure_function
            = lambda track : self.local_structure_function_power(track,6.)
            , further_conditioning =tracks_subfilters)


        plt.figure()
        plt.plot()

        plt.plot(S2.index/FPS,S2/mean_abs_sin_vel**2);
        l_coeff_S2 = plot_interpolant(S2/mean_abs_sin_vel**2,'b',slice_size=4)


        plt.plot(S4.index/FPS,S4/mean_abs_sin_vel**4,'r');
        l_coeff_S4 = plot_interpolant(S4/mean_abs_sin_vel**4,'r',slice_size=4)


        plt.plot(S6.index/FPS,S6/mean_abs_sin_vel**6,'g');
        l_coeff_S6 = plot_interpolant(S6/mean_abs_sin_vel**6,'g',slice_size=4)


        plt.xscale('log')
        plt.yscale('log')

        plt.grid()

        plt.legend(['S2'
                    , 'm=%g'%l_coeff_S2[0]
                    , 'samples lint'
                    , 'S4'
                    , 'm=%g'%l_coeff_S4[0]
                    , 'samples lint'
                    , 'S6'
                    , 'm=%g'%l_coeff_S6[0]
                    , 'samples lint'])

        plt.title('S2,S4,S6|(full tr,l<30) single (norm <v>^p =%g^p)'% mean_abs_sin_vel )



        plt.figure()


        slice_size = 100 #20
        slice_size_fit = 4

        plt.plot(S2[:slice_size],S4[:slice_size])
        l_coeff_st = plot_interpolant(S2[:slice_size]
                                   , 'g'
                                   , S4[:slice_size]
                                   , slice_size=slice_size_fit)

        slice_win = range(7,12)   
        l_coeff_md = plot_interpolant(S2[:slice_size]
                                   , 'r'
                                   , S4[:slice_size]
                                   , slice_size=slice_win)  


        plt.legend(['S2vS4','m=%g'%l_coeff_st[0],'samples lint' ,'m=%g'%l_coeff_md[0],'samples lint' ])




        plt.xscale('log')
        plt.yscale('log')

        plt.ylabel('S4')
        plt.xlabel('S2')

        plt.title('S2 vs. S4')
        plt.grid()





    # def structure_function(self):
    #     self.build_selection_criteria()
    #     self.tracks_with_atleast_one_pair = \
    #       self.trajs[self.crit_isolate_tracks_st_two_ped_happen_to_be_present]


    def single_ped_sqdv_analysis(self
                             , further_conditioning = None
                             , frames_interval = np.arange(1,100,3)#[1,2,20,30,40,50]
                                 ):
        
        if further_conditioning is None:
            tracks_subfilters = self.criteria['isolate_tracks_ped_is_always_alone'] 

        else:
            tracks_subfilters = self.criteria['isolate_tracks_ped_is_always_alone'] & further_conditioning
       
        effective_trajs = self.trajs[tracks_subfilters]


        plt.figure()
        plt.hist(effective_trajs['length'], bins=np.arange(0,100))

        plt.savefig('length_distrib.eps')

        self.single_ped_structure_function(plot_out=True
                                      , further_conditioning=further_conditioning
                                      , time_scale = 'frame')

        plt.savefig('S2 framebase.eps') 


        S2_gr = self.stats['single_ped_S2'].groupby('dt')
        

        for dt in frames_interval:
            plt.figure()
            plt.hist(S2_gr.get_group(dt)['sq|dv|'],log=True,bins=80)
            plt.title('Frame difference = %d' % dt )
            plt.xlabel('sq|dV|')
            plt.savefig('sqdv distribution frame_diff=%d.eps'% dt)
            plt.close()

        



    def multi_ped_structure_function(self
                                     ,further_conditioning = None):

        sel_frames_with_pairs = self.criteria['isolate_frames_with_pairs']

        ## indices of data structures of frame with pairs
        idx_of_pairs = self.datas.index[sel_frames_with_pairs]

        ## to have unique ids of pairs in the frames. 
        ## REM -> this ids are not unique. They must be coupled to the day
        ids_of_pairs = self.datas['data'][sel_frames_with_pairs].apply(lambda f : [p.id for p in f])

        ## just the sequence of days, to couple with the previous ones to have a well defined primary key
        day_idx = [idx[0] for idx in idx_of_pairs]


        
        ## further conditionings i.e. subfilters to the tracks
        #tracks_subfilters = self.crit_isolate_complete_tracks \
        #  & self.crit_isolate_tracks_shorter_than_N(30)

        


        ## trajectories ultimately considered
        if further_conditioning is None:
            effective_trajs = self.trajs
            tracks_subfilters = None

        else:
            tracks_subfilters = further_conditioning
            effective_trajs = self.trajs[tracks_subfilters]



        ## obtaining each group is a track -> every point will have same day and particle id.
        track_regrouping = groupby(zip(idx_of_pairs,ids_of_pairs), lambda x : [x[0][0],x[1]])
        ## every group will feature
        ## ((day,time),[id1 id2])

        groups = []
        uniquekeys = []

        for k, g in track_regrouping:
            groups.append(list(g))      # Store group iterator as a list
            uniquekeys.append(k)


        final_t_cof = []
        final_u_cof = []

        vel_cof = []

        final_t_ctf = []
        final_u_ctf = []

        vel_ctf = []



        for g in groups:
            tid1 = (g[0][0][0],g[0][1][0]) #this builds the two unique track ids
            tid2 = (g[0][0][0],g[0][1][1])

    #times = [gg[0][1] for gg in g]
    #print tid1,tid2

            try:
                t1 = effective_trajs.loc[tid1][ ['traj','u_median']]
                t2 = effective_trajs.loc[tid2][['traj','u_median']]
                track1,um1 = t1['traj'],t1['u_median']
                track2,um2 = t2['traj'],t2['u_median']
                # #print track1,track2

                out_u1,out_t1 = self.local_structure_function(track1)
                out_u2,out_t2 = self.local_structure_function(track2)
        
        #print out_u1
        
                if um1*um2 > 0: # cof
                    # print 'cof'
                    final_t_cof.extend(out_t1)
                    final_t_cof.extend(out_t2)
                    final_u_cof.extend(out_u1)
                    final_u_cof.extend(out_u2)

            #vel_cof.extend( list(np.sqrt(tp.generate_time_history(track1,tp.get_u)**2 \
                #                + tp.generate_time_history(track1,tp.get_v)**2)) )

                    vel_cof.extend ( list (tp.generate_velocity_speed_history(track1) ))
                    vel_cof.extend ( list (tp.generate_velocity_speed_history(track2) ))

            #vel_cof.extend( list(np.sqrt(tp.generate_time_history(track2,tp.get_u)**2 \
                #                + tp.generate_time_history(track2,tp.get_v)**2)) )

                else: #ctf
                    final_t_ctf.extend(out_t1)
                    final_t_ctf.extend(out_t2)
                    final_u_ctf.extend(out_u1)
                    final_u_ctf.extend(out_u2)


                    # vel_ctf.extend( list(np.sqrt(tp.generate_time_history(track1,tp.get_u)**2
                    #                 + tp.generate_time_history(track1,tp.get_v)**2)) )

                    # vel_ctf.extend( list(np.sqrt(tp.generate_time_history(track2,tp.get_u)**2
                    #                 + tp.generate_time_history(track2,tp.get_v)**2)) )


                    vel_ctf.extend ( list (tp.generate_velocity_speed_history(track1) ))
                    vel_ctf.extend ( list (tp.generate_velocity_speed_history(track2) ))

            except:
                print 'exc: ', g


        final_cof = pd.DataFrame(zip(final_t_cof,final_u_cof),columns=['dt','sq|dv|'])
        final_ctf = pd.DataFrame(zip(final_t_ctf,final_u_ctf),columns=['dt','sq|dv|'])
        self.single_ped_structure_function(further_conditioning = tracks_subfilters)
        final_sin = self.stats['single_ped_S2']


        cof_group = final_cof.groupby('dt')
        ctf_group = final_ctf.groupby('dt')
        sin_group = final_sin.groupby('dt')


        cof_means = final_cof.groupby('dt').mean()
        ctf_means = final_ctf.groupby('dt').mean()
        sin_means = final_sin.groupby('dt').mean()


        
        
        mean_abs_cof_vel = np.mean((vel_cof))
        mean_abs_ctf_vel = np.mean((vel_ctf))

        
        self.stats['mean_abs_sin_vel'] = np.mean([ s
                                     for t in self.tracks_with_always_singles['traj']
                                     for s in tp.generate_velocity_speed_history(t) ])

        

        self.stats['mean_abs_pairs_cof_vel'] = mean_abs_cof_vel
        self.stats['mean_abs_pairs_ctf_vel'] = mean_abs_ctf_vel
        mean_abs_sin_vel = self.stats['mean_abs_sin_vel'] 
        


        
        plt.figure()
        plt.plot(cof_means.index/FPS,cof_means/mean_abs_cof_vel**2,'b')
        l_coeff_cof = plot_interpolant(cof_means/mean_abs_cof_vel**2,'b')

        plt.plot(ctf_means.index/FPS,ctf_means/mean_abs_ctf_vel**2,'g')
        l_coeff_ctf = plot_interpolant(ctf_means/mean_abs_ctf_vel**2,'g')

        plt.plot(sin_means.index/FPS,sin_means/mean_abs_sin_vel**2,'r')
        l_coeff_sin = plot_interpolant(sin_means/mean_abs_sin_vel**2,'r')



        plt.xscale('log')
        plt.yscale('log')
        plt.grid()

        plt.title('S2')

        plt.legend(['cof,v=%f'% mean_abs_cof_vel
                    ,'m =%f'%l_coeff_cof[0]
                    , 'samples lint'
                    ,'ctf,v=%f'% mean_abs_ctf_vel
                    ,'m =%f'%l_coeff_ctf[0]
                    , 'samples lint'
                    ,'sin,v=%f'% mean_abs_sin_vel
                    ,'m =%f'%l_coeff_sin[0]
                    , 'samples lint'])



        
        plt.figure()
        plt.plot(cof_group.size().index/FPS,cof_group.size())
        plt.plot(ctf_group.size().index/FPS,ctf_group.size())
        plt.plot(sin_group.size().index/FPS,sin_group.size())

        plt.xscale('log')
        plt.yscale('log')

        plt.title('sample amount vs. dt')
        plt.xlabel('dt')
        plt.ylabel('#samples')
        plt.legend(['cof','ctf','sin'])

        plt.grid()

        plt.title('sample amount vs dt')






                
        

        


        


        
        

        

        

        


    
        
        
            




        


        
    


    

