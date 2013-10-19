import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import copy
from matplotlib.colors import LogNorm
from scipy.stats import maxwell
from scipy import interpolate
import scipy.ndimage.filters as filters
from scipy.stats import norm
from scipy.stats.mstats import normaltest
import matplotlib.mlab as mlab
import itertools

import json
parameters_glob = json.load(open('parameters.glob'))
#PATH_PREFIX='/home/acorbe/Documents/13.07.31/'
PATH_PREFIX='../results/'

av_bins = lambda bins : .5*(bins[1:]+bins[:-1])
sm_n = lambda n,sigma : filters.gaussian_filter(n,sigma)
smooth_hist = lambda n,bins,col,sigma : plt.plot(av_bins(bins),sm_n(n,sigma),color=col)


DEFAULT_SMOOTHING = parameters_glob['DEFAULT_SMOOTHING']
SPLINE_ORDER = parameters_glob['SPLINE_ORDER']
SPLINE_MIN_M = parameters_glob['SPLINE_MIN_M']
FPS = parameters_glob['FPS']
#DEFAULT_SMOOTHING = .005 # this is a good number for normal trajs

def set_x(te,val): te.x = val
def set_y(te,val): te.y = val
def set_z(te,val): te.z = val
def set_u(te,val): te.u = val
def set_v(te,val): te.v = val
def set_w(te,val): te.w = val
def set_ax(te,val): te.ax = val
def set_ay(te,val): te.ay = val
def set_az(te,val): te.az = val



get_x = lambda p: p.x
get_y = lambda p: p.y
get_z = lambda p: p.z

get_u = lambda p: p.u
get_v = lambda p: p.v
get_w = lambda p: p.w

get_ax = lambda p: p.ax
get_ay = lambda p: p.ay
get_az = lambda p: p.az

get_t = lambda p:p.t

SUB_SAMPLING_MINLENGTH = 2

def sub_sample_track(single_traj,sub_sampling_step,sub_sampling_offset = 0):
    if len(single_traj) > SUB_SAMPLING_MINLENGTH + sub_sampling_step:
        sub_sampling_offset = sub_sampling_offset % sub_sampling_step
        f_traj =  single_traj[ sub_sampling_offset::sub_sampling_step ]
    
        for p in ( p for i,p in enumerate(single_traj) if i not in xrange(sub_sampling_offset
                                                            , len(single_traj)
                                                            , sub_sampling_step) ):
        
            p.u = np.nan
            p.v = np.nan
            p.ax = np.nan
            p.ay = np.nan

        return f_traj

    else:
        return single_traj


def sub_sample_track_times(times,sub_sampling_step,sub_sampling_offset = 0):
    
    if len(times) > SUB_SAMPLING_MINLENGTH + sub_sampling_step:
        return times[ sub_sampling_offset::sub_sampling_step ]
    else:
        return times
            


def sub_sample_track_fps(single_traj,fps,sub_sampling_step):
    if len(single_traj) > SUB_SAMPLING_MINLENGTH + sub_sampling_step:
        return float(fps)/float(sub_sampling_step)
    else:
        return float(fps)
    
    
    


def generate_time_history(single_traj,f_rd):
    return np.asarray([f_rd(t) for t in single_traj])

def overwrite_time_history(single_traj,ntraj_history,f_wr):
    for p,val in zip(single_traj,ntraj_history):
        f_wr(p,val)



def generate_velocity_speed_history(single_traj):
    return np.sqrt( generate_time_history(single_traj, get_u )**2 \
                    + generate_time_history(single_traj, get_v )**2 )


def generate_spline_out_of_traj_RT(single_traj
                                   , f_rd
                                   , times
                                   , smoothing=DEFAULT_SMOOTHING
                                   , order=SPLINE_ORDER):
    t_val = np.asarray(times)
    x_val = np.asarray([f_rd(t) for t in single_traj])
    
    tck = interpolate.splrep(t_val,x_val, s=smoothing,k=order)
    return t_val,x_val,tck

    
                    
def generate_spline_out_of_traj(single_traj,f_rd,fps,smoothing = DEFAULT_SMOOTHING, order = SPLINE_ORDER):
    t_val = np.asarray([get_t(t) for t in single_traj])/fps #dependence on fps is here!!
    x_val = np.asarray([f_rd(t) for t in single_traj])
    
    tck = interpolate.splrep(t_val,x_val, s=smoothing, k=order)
    return t_val,x_val,tck

def perform_abstract_1Ddifferentiation_spline_based(single_traj
                                                    ,f_wr
                                                    ,f_rd
                                                    ,fps
                                                    ,order=1
                                                    ,smoothing=DEFAULT_SMOOTHING):
    #SPLINE_MIN_M = 3 
    if len(single_traj) > SPLINE_MIN_M:
        t_val,x_val,tck = generate_spline_out_of_traj(single_traj,f_rd,fps,smoothing=smoothing)
        xdot = interpolate.splev(t_val,tck,der=order)

        for xd,p in zip(xdot,single_traj):
            f_wr(p,xd)
    else:
        perform_abstract_1Ddifferentiation(single_traj,f_wr,f_rd,fps)



def perform_abstract_1Ddifferentiation_spline_based_RT(single_traj
                                                    ,f_wr
                                                    ,f_rd
                                                    ,times
                                                    ,order=1
                                                    ,smoothing=DEFAULT_SMOOTHING):
    #SPLINE_MIN_M = 3 
    if len(single_traj) > SPLINE_MIN_M:
        t_val,x_val,tck = generate_spline_out_of_traj_RT(single_traj
                                                         ,f_rd
                                                         ,times
                                                         ,smoothing=smoothing)
        xdot = interpolate.splev(t_val,tck,der=order)

        for xd,p in zip(xdot,single_traj):
            f_wr(p,xd)
    else:
        perform_abstract_1Ddifferentiation(single_traj,f_wr,f_rd,FPS)




def perform_abstract_1Ddifferentiation(single_traj,f_wr,f_rd,fps):
    """
    Generic tool to perform time differentiation along a trajectory.
    Input:

    single_traj - is a trajectory from openptv 'traj' data.

    f_wr - is a setter function for a trajectory element (u,v,ax,ay,...).

    f_rd - is a getter function for a trajectory element [can be a lambda]

    fps - time rate of the trajectory samples.


    EXAMPLE: determination of the x-component velocity
    call: perform_abstract_1Ddifferentiation(aTraj,usetter,xgetter,5)
    
    where:
    def usetter(Traj_element,val): Traj_element.u = val
    def xgetter(Traj_element): Traj_element.x         
    
    """
    f_wr(single_traj[-1] , (f_rd(single_traj[-1]) - f_rd(single_traj[-2]))*fps)
    f_wr(single_traj[0] ,  (f_rd(single_traj[1])- f_rd(single_traj[0]))*fps)

    for i,p in enumerate(single_traj[1:-1]):
        f_wr(p ,  (f_rd(single_traj[i+2]) - f_rd(single_traj[i]))/2.*fps)

        


def perform_abstract_1Ddifferentiation_fd_RT(single_traj,f_wr,f_rd,times):
    dt = np.diff(times)
    #print single_traj
    #print times
    #print "----------"
    
    f_wr(single_traj[-1] , (f_rd(single_traj[-1]) - f_rd(single_traj[-2]))/dt[0])
    f_wr(single_traj[0] ,  (f_rd(single_traj[1])- f_rd(single_traj[0]))/dt[-1])

    for i,p in enumerate(single_traj[1:-1]):
        f_wr(p ,  (f_rd(single_traj[i+2]) - f_rd(single_traj[i]))/(dt[i]+ dt[i+1]) )

    

def calculate_velocity(traj,fps):
    return calculate_velocity_fd_based(traj,fps)
    #return calculate_velocity_spline_based(traj,fps)


def calculate_acceleration(traj,fps):
    return calculate_acceleration_fd_based(traj,fps)
    #return calculate_acceleration_spline_based(traj,fps)


       
def calculate_velocity_fd_based(traj
                                , fps
                                , sub_sampling_step = 0):
    """ estimates velocity of the particles along the trajectory using simple
    forward and backward difference scheme
    Inputs:
        traj = Trajectory(), a list of Particle() objects linked in time
        fps = frame-per-second rate of recording, converts the frames to seconds
    Output: traj.{u,v,w} in meters/second    
    TODO: to be improved using smooth splines and higher order differentation 
    schemes
    """
    if sub_sampling_step == 0:    
        for t in traj: # for each trajectory
            perform_abstract_1Ddifferentiation(t, set_u, lambda te: te.x ,fps)
            perform_abstract_1Ddifferentiation(t, set_v, lambda te: te.y ,fps)
            perform_abstract_1Ddifferentiation(t, set_w, lambda te: te.z ,fps)
        
            # # last particle use backward difference
            # t[-1].u = (t[-1].x - t[-2].x)*fps # m/s
            # t[-1].v = (t[-1].y - t[-2].y)*fps
            # t[-1].w = (t[-1].z - t[-2].z)*fps
            # #first particle use forward difference
            # t[0].u = (t[1].x - t[0].x)*fps # m/s
            # t[0].v = (t[1].y - t[0].y)*fps
            # t[0].w = (t[1].z - t[0].z)*fps        
            # for i,p in enumerate(t[1:-1]): # for other particles..be aware of numberings!!
            #     p.u = (t[i+2].x - t[i].x)/2.*fps
            #     p.v = (t[i+2].y - t[i].y)/2.*fps
            #     p.w = (t[i+2].z - t[i].z)/2.*fps

    else:
        
        for t in traj:
            fps_s = sub_sample_track_fps(t,fps,sub_sampling_step)
            q = sub_sample_track(t,sub_sampling_step)
            perform_abstract_1Ddifferentiation(q, set_u, lambda te: te.x ,fps_s)
            perform_abstract_1Ddifferentiation(q, set_v, lambda te: te.y ,fps_s)
            perform_abstract_1Ddifferentiation(q, set_w, lambda te: te.z ,fps_s)
        
            
    
        
    
    return traj
    #return calculate_velocity_spline_based(traj,fps)

def calculate_velocity_fd_based_RT(traj
                                   , times
                                   , sub_sampling_step = 0):

    print "invoked!"

    if sub_sampling_step == 0:
        for t,time in itertools.izip(traj,times): # for each trajectory
            perform_abstract_1Ddifferentiation_fd_RT(t, set_u, lambda te: te.x ,time)
            perform_abstract_1Ddifferentiation_fd_RT(t, set_v, lambda te: te.y ,time)
            perform_abstract_1Ddifferentiation_fd_RT(t, set_w, lambda te: te.z ,time)

    else:        
        for t,time in itertools.izip(traj,times): # for each trajectory
            q = sub_sample_track(t,sub_sampling_step)
            qtime = sub_sample_track_times(time,sub_sampling_step)
            perform_abstract_1Ddifferentiation_fd_RT(q, set_u, lambda te: te.x ,qtime)
            perform_abstract_1Ddifferentiation_fd_RT(q, set_v, lambda te: te.y ,qtime)
            perform_abstract_1Ddifferentiation_fd_RT(q, set_w, lambda te: te.z ,qtime)
        
    return traj


    

def calculate_velocity_spline_based(traj
                                    , fps
                                    , smoothing=DEFAULT_SMOOTHING
                                    , sub_sampling_step = 0):

    if sub_sampling_step == 0:
        for t in traj:
            perform_abstract_1Ddifferentiation_spline_based(t, set_u, lambda te: te.x ,fps,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(t, set_v, lambda te: te.y ,fps,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(t, set_w, lambda te: te.z ,fps,smoothing=smoothing)
    else:
        
        for t in traj:
            fps_s = sub_sample_track_fps(t,fps,sub_sampling_step)
            q = sub_sample_track(t,sub_sampling_step)
            
            perform_abstract_1Ddifferentiation_spline_based(q, set_u, lambda te: te.x ,fps_s,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(q, set_v, lambda te: te.y ,fps_s,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(q, set_w, lambda te: te.z ,fps_s,smoothing=smoothing)            
        

    return traj


def calculate_velocity_spline_based_RT(traj,times,smoothing=DEFAULT_SMOOTHING, sub_sampling_step = 0):

    if sub_sampling_step == 0:    
        for t,time in itertools.izip(traj,times):
            perform_abstract_1Ddifferentiation_spline_based_RT(t, set_u, lambda te: te.x ,time,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(t, set_v, lambda te: te.y ,time,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(t, set_w, lambda te: te.z ,time,smoothing=smoothing)

    else:
        for t,time in itertools.izip(traj,times):
            q = sub_sample_track(t,sub_sampling_step)
            qtime = sub_sample_track_times(time,sub_sampling_step)
            perform_abstract_1Ddifferentiation_spline_based_RT(q, set_u, lambda te: te.x ,qtime,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(q, set_v, lambda te: te.y ,qtime,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(q, set_w, lambda te: te.z ,qtime,smoothing=smoothing)


    return traj
        
    
def calculate_acceleration_fd_based(traj,fps,sub_sampling_step = 0):
    """ estimates acceleration of the person along the trajectory using simple
    forward and backward difference scheme
    Inputs:
        traj = Trajectory(), a list of Particle() objects linked in time
        fps = frame-per-second rate of recording, converts the frames to seconds
    Output: traj.{ax,ay,az} in meters/second**2    
    TODO: to be improved using smooth splines and higher order differentation 
    schemes
    """
    if sub_sampling_step == 0:    
        for t in traj: # for each trajectory
            perform_abstract_1Ddifferentiation(t, set_ax, lambda te: te.u ,fps)
            perform_abstract_1Ddifferentiation(t, set_ay, lambda te: te.v ,fps)
            perform_abstract_1Ddifferentiation(t, set_az, lambda te: te.w ,fps)
    else:
        
        for t in traj:
            fps_s = sub_sample_track_fps(t,fps,sub_sampling_step)
            q = sub_sample_track(t,sub_sampling_step)
            perform_abstract_1Ddifferentiation(q, set_ax, lambda te: te.u ,fps_s)
            perform_abstract_1Ddifferentiation(q, set_ay, lambda te: te.v ,fps_s)
            perform_abstract_1Ddifferentiation(q, set_az, lambda te: te.w ,fps_s)
    
    return traj

    #return calculate_acceleration_spline_based(traj,fps)

def calculate_acceleration_fd_based_RT(traj,times,sub_sampling_step = 0):    
    if sub_sampling_step == 0:
        for t,time in itertools.izip(traj,times): # for each trajectory
            perform_abstract_1Ddifferentiation_fd_RT(t, set_ax, lambda te: te.u ,time)
            perform_abstract_1Ddifferentiation_fd_RT(t, set_ay, lambda te: te.v ,time)
            perform_abstract_1Ddifferentiation_fd_RT(t, set_az, lambda te: te.w ,time)

    else:
        for t,time in itertools.izip(traj,times): # for each trajectory
            q = sub_sample_track(t,sub_sampling_step)
            qtime = sub_sample_track_times(time,sub_sampling_step)
            perform_abstract_1Ddifferentiation_fd_RT(q, set_ax, lambda te: te.u ,qtime)
            perform_abstract_1Ddifferentiation_fd_RT(q, set_ay, lambda te: te.v ,qtime)
            perform_abstract_1Ddifferentiation_fd_RT(q, set_az, lambda te: te.w ,qtime)

        
    return traj


    

def calculate_acceleration_spline_based(traj,fps,smoothing=DEFAULT_SMOOTHING,sub_sampling_step = 0):
    if sub_sampling_step == 0:
        for t in traj:  
            perform_abstract_1Ddifferentiation_spline_based(t, set_ax, lambda te: te.x ,fps,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(t, set_ay, lambda te: te.y ,fps,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(t, set_az, lambda te: te.z ,fps,order=2,smoothing=smoothing)

    else:
        
        for t in traj:
            fps_s = sub_sample_track_fps(t,fps,sub_sampling_step)
            q = sub_sample_track(t,sub_sampling_step)
            perform_abstract_1Ddifferentiation_spline_based(q, set_ax, lambda te: te.x ,fps_s,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(q, set_ay, lambda te: te.y ,fps_s,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based(q, set_az, lambda te: te.z ,fps_s,order=2,smoothing=smoothing)


    return traj



def calculate_acceleration_spline_based_RT(traj,times,smoothing=DEFAULT_SMOOTHING, sub_sampling_step = 0):

    if sub_sampling_step == 0:
        for t,time in itertools.izip(traj,times):
            perform_abstract_1Ddifferentiation_spline_based_RT(t, set_ax, lambda te: te.x ,time,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(t, set_ay, lambda te: te.y ,time,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(t, set_az, lambda te: te.z ,time,order=2,smoothing=smoothing)

    else:
        for t,time in itertools.izip(traj,times):
            q = sub_sample_track(t,sub_sampling_step)
            qtime = sub_sample_track_times(time,sub_sampling_step)
            perform_abstract_1Ddifferentiation_spline_based_RT(q, set_ax, lambda te: te.x ,qtime,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(q, set_ay, lambda te: te.y ,qtime,order=2,smoothing=smoothing)
            perform_abstract_1Ddifferentiation_spline_based_RT(q, set_az, lambda te: te.z ,qtime,order=2,smoothing=smoothing)


    return traj



def query_track(track,getter):
    ret = np.array([getter(p) for p in track])
    return ret



####
#
# CLASS IS OBSOLETE - REPLACED BY CLASS IN TRAJ_CORRELATION.PY
# THIS IS KEPT FOR SOME STATIC METHODS TO BE REFACTORED
#
#
####
class Trajectory_stats(object):
    '''this class contains statistics on trajectories such as initial and ending point,
    max acceleration, max velocity, track length and so on.'''
    
    def __init__(self):

        self.start_x = [] #X1
        self.start_y = [] #X2
        self.end_x = [] #X3
        self.end_y = [] #X4
        
        self.any_ax = [] #A1
        self.any_ay = [] #A2
        self.max_a = []  #A3
        self.any_a = []  #A4
        

        self.any_vx = [] #V1
        self.any_vy = [] #V2
        self.max_v = []  #V3       
        self.any_v = []  #V4   
        self.median_vx = [] #V5

        
        self.track_length=[] #L1

        #self.how_many_others=[] #Q1


    # def qDirection_list(self,track_id):
    #     #fin_ids.append(track_id)
    #     direction = [ self.median_vx[id_] for id_ in track_id ]
    #     return direction

    
    def qDirection(self,track_id):
        #fin_ids.append(track_id)
        ##direction = [ self.median_vx[id_] for id_ in track_id ]
        return self.median_vx[track_id]

    
    def el2nparray(self,el):
        el = np.array(el)
        
    def calculate_traj_stats(self,traj):
 
        for idx,t in enumerate(traj):
            #Position part
            self.start_x.append( t[0].x ) #X1
            self.start_y.append( t[0].y ) #X2
            self.end_x.append( t[-1].x ) #X3
            self.end_y.append( t[-1].y ) #X4

                    
            #Acceleration part
            ax = query_track(t,get_ax) #A1
            self.any_ax.append(ax)

            ay = query_track(t,get_ay) #A2
            self.any_ay.append(ay)
                      
            cur_a = np.sqrt(ax**2 + ay**2) #A3
            self.any_a.append(cur_a)
 
            self.max_a.append(np.max(cur_a)) #A4

            #Velocity part
            vx = query_track(t,get_u) #V1
            self.any_vx.append(vx)

            vy = query_track(t,get_v) #V2
            self.any_vy.append(vy)

            cur_v = np.sqrt(vx**2 + vy**2) #V3
            self.any_v.append(cur_v)

            self.max_v.append(np.max(cur_v)) #V4

            self.median_vx.append(np.median(vx)) #V5


            #Track length part
            self.track_length.append(len(t)) #L1

            

        self.start_x = np.asarray(self.start_x) #X1
        self.start_y = np.asarray(self.start_y) #X2
        self.end_x = np.asarray(self.end_x) #X3
        self.end_y = np.asarray(self.end_y) #X4

            
        self.any_ax = np.asarray(self.any_ax)  #A1
        self.any_ay = np.asarray(self.any_ay)  #A2
        self.max_a = np.asarray(self.max_a)    #A3
        self.any_a = np.asarray(self.any_a)    #A4

        
        self.any_vx = np.asarray(self.any_vx)  #V1
        self.any_vy = np.asarray(self.any_vy)  #V2
        self.max_v = np.asarray(self.max_v)    #V3
        self.any_v = np.asarray(self.any_v)    #V4
        #self.median_v = np.asarray(self.any_v)    #V5
        
        
        self.track_length = np.asarray(self.track_length) #L1


    def accumulate_traj(self,traj):
 
        for idx,t in enumerate(traj):
            #Position part
            self.start_x.append( t[0].x ) #X1
            self.start_y.append( t[0].y ) #X2
            self.end_x.append( t[-1].x ) #X3
            self.end_y.append( t[-1].y ) #X4

                    
            #Acceleration part
            ax = query_track(t,get_ax) #A1
            self.any_ax.append(ax)

            ay = query_track(t,get_ay) #A2
            self.any_ay.append(ay)
                      
            cur_a = np.sqrt(ax**2 + ay**2) #A3
            self.any_a.append(cur_a)
 
            self.max_a.append(np.max(cur_a)) #A4

            #Velocity part
            vx = query_track(t,get_u) #V1
            self.any_vx.append(vx)

            vy = query_track(t,get_v) #V2
            self.any_vy.append(vy)

            cur_v = np.sqrt(vx**2 + vy**2) #V3
            self.any_v.append(cur_v)

            self.max_v.append(np.max(cur_v)) #V4

            self.median_vx.append(np.median(vx)) #V5


            #Track length part
            self.track_length.append(len(t)) #L1

            
    def compute_traj_stats(self):
        self.start_x = np.asarray(self.start_x) #X1
        self.start_y = np.asarray(self.start_y) #X2
        self.end_x = np.asarray(self.end_x) #X3
        self.end_y = np.asarray(self.end_y) #X4

            
        self.any_ax = np.asarray(self.any_ax)  #A1
        self.any_ay = np.asarray(self.any_ay)  #A2
        self.max_a = np.asarray(self.max_a)    #A3
        self.any_a = np.asarray(self.any_a)    #A4

        
        self.any_vx = np.asarray(self.any_vx)  #V1
        self.any_vy = np.asarray(self.any_vy)  #V2
        self.max_v = np.asarray(self.max_v)    #V3
        self.any_v = np.asarray(self.any_v)    #V4
        self.median_v = np.asarray(self.any_v)    #V5

        
        
        
        self.track_length = np.asarray(self.track_length) #L1
    
        
    def hist_track_length(self):
        plt.figure()
        plt.hist(self.track_length,log=True,bins=80)
        plt.title("track length, not normed!")    
    
    def hist_max_a(self):
        plt.figure()
        plt.hist(self.max_a,normed=True,log=True,bins=80)
        plt.title("max acc distribution") 
    

    def hist_joint_start_end_distrib(self):
        plt.figure()
        plt.hist2d(self.start_x,self.end_x,bins=20)
        plt.colorbar()
        plt.xlabel('x [m]')
        plt.ylabel('x [m]')
        plt.title('joint distribution (traj.start.x,traj.end.x)')





def purge_bad_tracks(traj,bad_track_idx):
    #print bad_track_idx
    new_arr = np.delete(traj, bad_track_idx)
    return new_arr

def ref_to_np_array(cl,qtys):
    for qty_str in qtys:
        qty = getattr(cl,qty_str)
        setattr(cl,qty_str,np.array(qty))

class LagrangianStats():
    """Handles lagrangian like statististics. Contains several accumulators and
    relative post processing functions.
    Do help(LagrangianStats.hist*) to see more about the post processing functions.
    """
    def __init__(self):
        """
        In the description of the following containers the '|' will denote the
        conditional probability.
        """

        
        ### Containers for conditional individual position/velocity/acceleration ###
        #every x position in frames
        self.tot_x,\
          self.tot_y = [],[]

        #every x position in frames | 1 ped at once in frame
        self.single_x,\
          self.single_y = [],[]

        #every x position in frames | (1 ped at once in frame, velocity is mostly rightly oriented)
        self.single_x_gt,\
            self.single_x_lt,\
            self.single_y_gt,\
            self.single_y_lt = [],[],[],[]

        #every u velocity in frames | 1 ped at once
        self.single_u,\
          self.single_v,\
          self.single_ax,\
          self.single_ay = [],[],[],[]

        self.single_speed = []
        self.single_acc = []

        #every u velocity in frames | 1 ped at once, mostly right trajectory
        self.single_u_gt,\
            self.single_u_lt,\
            self.single_v_gt,\
            self.single_v_lt = [],[],[],[]
        self.single_ax_gt,\
            self.single_ax_lt,\
            self.single_ay_gt,\
            self.single_ay_lt = [],[],[],[]

        #every u velocity in frames | 1 or more ped, coflow
        self.multi_u_cof,\
            self.multi_u_ctf,\
            self.multi_v_cof,\
            self.multi_v_ctf = [],[],[],[]

        self.multi_speed_cof,\
          self.multi_speed_ctf = [],[]
            
        self.multi_ax_cof,\
            self.multi_ax_ctf,\
            self.multi_ay_cof,\
            self.multi_ay_ctf = [],[],[],[]

        self.multi_acc_cof,\
          self.multi_acc_ctf = [],[]

            
        ### Containers for orbitwise/global direction ###
        #amount of traj going to the right
        self.going_to_the_right,\
          self.going_to_the_left  = [],[]

        ### Containers for average frame properties ###
        #amount of pedestrians in a frame
        self.presence = []

        #amount of pedestrians in a frame | coflow
        self.presence_cof,\
          self.presence_ctf = [],[]

        self.framewise_mean_abs_vel,\
            self.framewise_mean_abs_vel_cof,\
            self.framewise_mean_abs_vel_ctf = [],[],[]

        self.framewise_abs_vel,\
          self.framewise_abs_vel_cof,\
          self.framewise_abs_vel_ctf = [],[],[]

            
        ### Containers for pairwise properties ###
        #pairwise x-distance for every pair of ped in frame
        self.dist_x,\
            self.dist_y,\
            self.dist,\
            self.delta_vx,\
            self.delta_vy,\
            self.delta_v = [],[],[],[],[],[]

        #pairwise x-distance for every pair of ped in frame | frame is in coflow
        self.dist_x_cof,\
            self.dist_y_cof,\
            self.dist_cof,\
            self.delta_vx_cof,\
            self.delta_vy_cof,\
            self.delta_v_cof = [],[],[],[],[],[]

        #pairwise x-distance for every pair of ped in frame | frame is in counterflow
        self.dist_x_ctf,\
            self.dist_y_ctf,\
            self.dist_ctf,\
            self.delta_vx_ctf,\
            self.delta_vy_ctf,\
            self.delta_v_ctf = [],[],[],[],[],[]



        #standard parameters for histograms
        self.hist_param = {'bins':80 , 'normed':True, 'alpha':.5, 'histtype':'step', 'log':True}
        self.hist_param_noNormed = {'bins':80 , 'alpha':.5, 'histtype':'step', 'log':True}
        self.hist_param_noLog = {'bins':80 , 'normed':True, 'alpha':.5, 'histtype':'step'}
        self.hist2d_param = {'bins':80,'normed':True}
        

    def accumulate_lagrangian_stats(self,data,traj,id2TrajMap,traj_stats):
        self.going_to_the_right.extend(\
            [traj_stats.qDirection(t)  for t in range(len(traj))\
                 if traj_stats.qDirection(t) > 0 ])

        self.going_to_the_left.extend(\
             [traj_stats.qDirection(t)  for t in range(len(traj))\
                  if traj_stats.qDirection(t) < 0 ])

        
        for f in data:
            frame_average_vel,frame_any_velocity =\
              self.sub_accumulate_lagrangian_stats_eval_avg_frame_vel(f)

            self.framewise_mean_abs_vel.append(frame_average_vel)
            self.framewise_abs_vel.extend(frame_any_velocity)
            
            if len(f) == 1: # single person frame
                cur_id = f[0].id
                #print cur_id
                if traj_stats.qDirection(id2TrajMap[cur_id]) > 0:
                    self.single_u_gt.append(f[0].u)
                    self.single_v_gt.append(f[0].v)
                    self.single_ax_gt.append(f[0].ax)
                    self.single_ay_gt.append(f[0].ay)

                    self.single_x_gt.append(f[0].x)
                    self.single_y_gt.append(f[0].y)
                                        
                else:
                    self.single_u_lt.append(f[0].u)
                    self.single_v_lt.append(f[0].v)
                    self.single_ax_lt.append(f[0].ax)
                    self.single_ay_lt.append(f[0].ay)
                    
                    self.single_x_lt.append(f[0].x)
                    self.single_y_lt.append(f[0].y)
                    

                self.single_u.append(f[0].u)
                self.single_v.append(f[0].v)

                
                
                self.single_ax.append(f[0].ax)
                self.single_ay.append(f[0].ay)
                
                self.single_x.append(f[0].x)
                self.single_y.append(f[0].y)
                self.tot_x.append(f[0].x)
                self.tot_y.append(f[0].y)
                
                self.presence.append(1)
                self.presence_cof.append(1)
                self.presence_ctf.append(1)

                self.framewise_mean_abs_vel_cof.append(frame_average_vel)
                self.framewise_mean_abs_vel_ctf.append(frame_average_vel)
                self.framewise_abs_vel_cof.extend(frame_any_velocity)
                self.framewise_abs_vel_ctf.extend(frame_any_velocity)

                                    
            else:
                all_u = [p.u for p in f]
                #print all_u
                all_v = [p.v for p in f]
                all_ax = [p.ax for p in f]
                all_ay = [p.ay for p in f]

                dist_x,dist_y,dist,delta_vx,delta_vy,delta_v =\
                        self.sub_accumulate_lagrangian_stats_pair_processing(f)

                self.sub_extend_pair_stats_all(dist_x,dist_y,dist,delta_vx,delta_vy,delta_v)

                self.presence.append(len(f))

                if np.all(np.array(all_u) > 0) or np.all(np.array(all_u) < 0 ):
                    self.multi_u_cof.extend(all_u)
                    self.multi_v_cof.extend(all_v)
                    self.multi_ax_cof.extend(all_ax)
                    self.multi_ay_cof.extend(all_ay)
                    
                    self.sub_extend_pair_stats_cof(dist_x,dist_y,dist,delta_vx,delta_vy,delta_v)
                    self.presence_cof.append(len(f))

                    self.framewise_mean_abs_vel_cof.append(frame_average_vel)
                    self.framewise_abs_vel_cof.extend(frame_any_velocity)
                    
                else:
                    #print "ctf"
                    self.multi_u_ctf.extend(all_u)
                    self.multi_v_ctf.extend(all_v)
                    self.multi_ax_ctf.extend(all_ax)
                    self.multi_ay_ctf.extend(all_ay)

                    self.sub_extend_pair_stats_ctf(dist_x,dist_y,dist,delta_vx,delta_vy,delta_v)
                    self.presence_ctf.append(len(f))

                    self.framewise_mean_abs_vel_ctf.append(frame_average_vel)
                    self.framewise_abs_vel_ctf.extend(frame_any_velocity)

                self.tot_x.extend([p.x for p in f ])
                self.tot_y.extend([p.y for p in f ])



                
                


    def group_in_all_stats(self):
        self.all_stats = {'single':[],'multi':[]}
        self.all_stats['single'] = {\
            'x':{'all':self.single_x , 'gt':self.single_x_gt,'lt':self.single_x_lt},\
            'y':{'all':self.single_y , 'gt':self.single_y_gt,'lt':self.single_y_lt},\
            'u':{'all':self.single_u , 'gt':self.single_u_gt,'lt':self.single_u_lt},\
            'v':{'all':self.single_v , 'gt':self.single_v_gt,'lt':self.single_v_lt},\
            'ax':{'all':self.single_ax , 'gt':self.single_ax_gt,'lt':self.single_ax_lt},\
            'ay':{'all':self.single_ay , 'gt':self.single_ay_gt,'lt':self.single_ay_lt},\
        }

        self.single_speed = np.sqrt(np.asarray(self.single_u)**2 \
                                  +  np.asarray(self.single_v)**2)

        self.single_acc = np.sqrt(np.asarray(self.single_ax)**2 \
                                  + np.asarray(self.single_ay)**2)



        self.multi_speed_ctf = np.sqrt(np.asarray(self.multi_u_ctf)**2 \
                                  +  np.asarray(self.multi_v_ctf)**2)

        self.multi_speed_cof = np.sqrt(np.asarray(self.multi_u_cof)**2 \
                                  +np.asarray(self.multi_v_cof)**2)


                                  
        self.multi_acc_ctf = np.sqrt(np.asarray(self.multi_ax_ctf)**2 \
                                  +  np.asarray(self.multi_ay_ctf)**2)

        self.multi_acc_cof = np.sqrt(np.asarray(self.multi_ax_cof)**2 \
                                  +np.asarray(self.multi_ay_cof)**2)



        # ref_to_np_array(self,["tot_x","tot_y"\
        #                   , "single_x", "single_y"\
        #                   , "single_x_gt", "single_x_lt", "single_y_gt", "single_y_lt"\
        #                   , "single_u", "single_v", "single_ax","single_ay"\
        #                   , "single_speed" , "single_acc"\
        #                   , "single_u_gt","single_u_lt","single_v_gt","single_v_lt"
        #                   , "single_ax_gt","single_ax_lt","single_ay_gt","single_ay_lt"
        #                   , "multi_u_cof","multi_u_ctf","multi_v_cof","multi_v_ctf"
        #                   , "multi_speed_cof", "multi_speed_ctf"
        #                   , "multi_ax_cof","multi_ax_ctf","multi_ay_cof","multi_ay_ctf"
        #                   , "multi_acc_cof", "multi_acc_ctf"
        #                   , "going_to_the_right", "going_to_the_left"
        #                   , "presence", "presence_cof" , "presence_ctf"
        #                   , "framewise_mean_abs_vel" , "framewise_mean_abs_vel_cof" , "framewise_mean_abs_vel_ctf"
        #                   , "framewise_abs_vel", "framewise_abs_vel_cof" , "framewise_mean_abs_vel_ctf"
        #                   , "dist_x" , "dist_y" , "dist" , "delta_vx" , "delta_vy" , "delta_v"
        #                   , "dist_x_cof" , "dist_y_cof" , "dist_cof" , "delta_vx_cof" , "delta_vy_cof" , "delta_v_cof"
        #                   , "dist_x_ctf" , "dist_y_ctf" , "dist_ctf" , "delta_vx_ctf" , "delta_vy_ctf" , "delta_v_ctf"
        #                   ])


                                    
                

    def sub_accumulate_lagrangian_stats_eval_avg_frame_vel(self,frame):
        speeds_in_frame = [np.sqrt(p.u**2 + p.v**2) for p in frame] 
        return np.mean(speeds_in_frame),speeds_in_frame

    
    def sub_accumulate_lagrangian_stats_pair_processing(self,frame):
        gen_pairs = lambda : it.combinations(frame,2) 
        pairs = gen_pairs()
        dist_x = [np.abs(p1.x - p2.x) for p1,p2 in pairs]
        
        pairs = gen_pairs()
        dist_y = [np.abs(p1.y - p2.y) for p1,p2 in pairs]
        
        pairs = gen_pairs()
        dist = [np.sqrt(dx**2 + dy**2) for dx,dy in zip(dist_x,dist_y)]

        pairs = gen_pairs()
        delta_vx = [np.abs(p1.u - p2.u) for p1,p2 in pairs]
        
        pairs = gen_pairs()
        delta_vy = [np.abs(p1.v - p2.v) for p1,p2 in pairs]
        
        pairs = gen_pairs()
        delta_v  = [np.sqrt(dvx**2 + dvy**2) \
                        for dvx,dvy in zip(delta_vx,delta_vy)]


        return (dist_x,dist_y,dist,delta_vx,delta_vy,delta_v)


    def sub_extend_pair_stats_all(self,dist_x,dist_y,dist,delta_vx,delta_vy,delta_v ):
        self.dist_x.extend(dist_x)
        self.dist_y.extend(dist_y)
        self.dist.extend(dist)
        self.delta_vx.extend(delta_vx) 
        self.delta_vy.extend(delta_vy)
        self.delta_v.extend(delta_v)
    
    def sub_extend_pair_stats_ctf(self,dist_x,dist_y,dist,delta_vx,delta_vy,delta_v ):
        self.dist_x_ctf.extend(dist_x)
        self.dist_y_ctf.extend(dist_y)
        self.dist_ctf.extend(dist)
        self.delta_vx_ctf.extend(delta_vx) 
        self.delta_vy_ctf.extend(delta_vy)
        self.delta_v_ctf.extend(delta_v)
    
    def sub_extend_pair_stats_cof(self,dist_x,dist_y,dist,delta_vx,delta_vy,delta_v ):
        self.dist_x_cof.extend(dist_x)
        self.dist_y_cof.extend(dist_y)
        self.dist_cof.extend(dist)
        self.delta_vx_cof.extend(delta_vx) 
        self.delta_vy_cof.extend(delta_vy)
        self.delta_v_cof.extend(delta_v)

    def sub_eval_desired_velocity_qtys(self\
                                ,grid_x=40,grid_y=40,small_incr=.00001\
                                ,qty='vel'
                                ,spec_type='gt'):
                                #,img_format='png'
                                #,**kw):
                                

        #spec_type = 'lt'

        if qty == 'vel':
            field_x = 'u'
            field_y = 'v'
        elif qty == 'acc':
            field_x = 'ax'
            field_y = 'ay'
                
        
        x_val = self.all_stats['single']['x'][spec_type]
        y_val = self.all_stats['single']['y'][spec_type]
        u_val = self.all_stats['single'][field_x][spec_type]
        v_val = self.all_stats['single'][field_y][spec_type]
        
        spatial_box = {'x_max':np.max(x_val), 'x_min':np.min(x_val),\
            'y_max':np.max(y_val), 'y_min':np.min(y_val)}

        #plt.quiver(self.single_x_gt,self.single_y_gt,self.single_u_gt,self.single_v_gt,scale=50)


        find_class = lambda v, v_min, v_max, tot: np.floor( (v - v_min)/(v_max - v_min) * tot )

        
        find_class_x = lambda x : find_class(x,spatial_box['x_min']-small_incr,spatial_box['x_max'] + small_incr,grid_x)
        find_class_y = lambda y : find_class(y,spatial_box['y_min']-small_incr,spatial_box['y_max'] + small_incr,grid_y)
        
        u_all = [ [] for i in range(grid_x*grid_y)]
        v_all = [ [] for i in range(grid_x*grid_y)]

        cl_x_back = [ 0 for i in range(grid_x*grid_y)]
        cl_y_back = [ 0 for i in range(grid_x*grid_y)]

        x_class_v = [ 0 for i in range(grid_x*grid_y)]
        y_class_v = [ 0 for i in range(grid_x*grid_y)]

        
        x_min = spatial_box['x_min']
        x_max = spatial_box['x_max']
        y_min = spatial_box['y_min']
        y_max = spatial_box['y_max']
        
                        
        for cl_x in range(grid_x):
            for cl_y in range(grid_y):
                cl_tot = int(cl_x + cl_y*grid_x)

                cl_x_back[cl_tot] = cl_x
                cl_y_back[cl_tot] = cl_y

                x_class_v[cl_tot] = x_min - small_incr + (x_max - x_min + 2 * small_incr)*(float(cl_x)/float(grid_x) + .5)
                y_class_v[cl_tot] = y_min - small_incr + (y_max - y_min + 2 * small_incr)*(float(cl_y)/float(grid_y) + .5)

        for x,y,u,v in zip(x_val,y_val,u_val,v_val):
            cl_x = find_class_x(x)
            cl_y = find_class_y(y)

            cl_tot = int(cl_x + cl_y*grid_x)
            u_all[cl_tot].append(u)
            v_all[cl_tot].append(v)




        v_all = [vs for vs,us  in zip(v_all,u_all) if not us == [] ]
        x_class_v = [vs for vs,us  in zip(x_class_v,u_all) if not us == [] ]
        y_class_v = [vs for vs,us  in zip(y_class_v,u_all) if not us == [] ]
        u_all = [us for us in u_all if not us == [] ]

        u_means = [np.mean(us) for us in u_all]
        #u_delta_check = [np.mean( np.asarray(us) - u_mean ) for us , u_mean in zip(u_all,u_means) ]
        #u_delta = [u_val_ - u_mean for u_val_ in us for us , u_mean in zip(u_all,u_means)]
        

        v_means = [np.mean(vs) for vs in v_all]
        #v_delta_check = [np.mean( np.asarray(vs) - v_mean ) for vs , v_mean in zip(v_all,v_means) ]
        #v_delta = [v_val_ - v_mean for v_val_ in vs  for vs , v_mean in zip(v_all,v_means)]

        u_delta = []
        v_delta = []
        for us,vs, u_mean,v_mean in zip(u_all,v_all,u_means,v_means):
            for u_,v_ in zip(us,vs):
                u_delta.append(u_ - u_mean)
                v_delta.append(v_ - v_mean)
        
        
    
        #plt.figure()
        #plt.hist(u_delta_check,**self.hist_param)
        #plt.hist(v_delta_check,**self.hist_param)

        return x_class_v,y_class_v,u_means,v_means,u_delta,v_delta,u_all,v_all

    def quiver_desired_velocity(self\
                                ,grid_x=40,grid_y=40,small_incr=.00001\
                                ,qty='vel'
                                ,spec_type='gt'
                                ,img_format='png'
                                ,**kw):


        x_class_v,y_class_v,u_means,v_means,u_delta,v_delta,u_all,v_all =\
           self.sub_eval_desired_velocity_qtys(\
                                               grid_x,grid_y,small_incr\
                                               ,qty\
                                               ,spec_type)


        plt.figure()
        plt.hist2d(u_delta,v_delta,**self.hist2d_param)
        plt.title('Overall noise distribution')
        plt.savefig(PATH_PREFIX + 'total_noise_%s_%s.%s' % (qty,spec_type,img_format))


        rv = maxwell()

        speed_delta = np.sqrt(np.asarray(u_delta)**2 + np.asarray(v_delta)**2)

        loc,scale = maxwell.fit(speed_delta)
        
        plt.figure()
        plt.hist(speed_delta,**self.hist_param_noLog)
        x = np.linspace(0,2,100)
        plt.plot(x,maxwell.pdf(x,scale=scale,loc=loc))
        
        plt.title('Overall abs noise distribution')
        plt.savefig(PATH_PREFIX + 'maxwellian_noise_%s_%s.%s' % (qty,spec_type,img_format))

        
        temperature  = np.sqrt( np.asarray([np.var(us) for us in u_all]) + np.asarray([np.var(vs) for vs in v_all]) )

        speed = [np.mean(np.sqrt(np.asarray(us)**2 + np.asarray(vs)**2)) for us,vs in zip(u_all,v_all) ]

        plt.figure()
        plt.quiver(x_class_v,y_class_v,u_means,v_means,temperature,**kw)
        plt.colorbar()
        plt.title('cross-realization "thermal" noise [m/s]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.savefig(PATH_PREFIX + 'quiver_thermal_noise_%s_%s.%s' % (qty,spec_type,img_format))
        
        plt.figure()
        plt.quiver(x_class_v,y_class_v,u_means,v_means,speed,**kw)
        plt.colorbar()
        plt.title('average speed')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.savefig(PATH_PREFIX + 'quiver_average_modulus_%s_%s.%s' % (qty,spec_type,img_format))


        #return shape,loc

    def hist_desired_speed_temperature(self\
                                ,grid_x=40,grid_y=40,small_incr=.00001\
                                ,qty='vel'
                                ,img_format='png'):
                                

    
        x_class_v_g,y_class_v_g,u_means_g,v_means_g,u_delta_g,v_delta_g,u_all_g,v_all_g =\
           self.sub_eval_desired_velocity_qtys(\
                                               grid_x,grid_y,small_incr\
                                               ,qty\
                                               ,spec_type='gt')
      

        speed_delta_g = np.sqrt(np.asarray(u_delta_g)**2 + np.asarray(v_delta_g)**2)


        x_class_v_g,y_class_v_g,u_means_g,v_means_g,u_delta_g,v_delta_g,u_all_g,v_all_g =\
           self.sub_eval_desired_velocity_qtys(\
                                               grid_x,grid_y,small_incr\
                                               ,qty\
                                               ,spec_type='lt')

       

        speed_delta_l = np.sqrt(np.asarray(u_delta_g)**2 + np.asarray(v_delta_g)**2)

        plt.figure()
        
        plt.hist(speed_delta_g,**self.hist_param_noLog)
        plt.hist(speed_delta_l,**self.hist_param_noLog)
        
        plt.title('Speed fluctuation comparison')
        plt.legend(['2R','2L'])

        plt.savefig(PATH_PREFIX + 'speed_delta_comp_%s.%s' % (qty,img_format))

       
 
    def hist_discrete_funddiag(self,factor=1.,exponent=.1,exp2=.75):
        plt.figure()
        
        H_any = np.bincount(self.presence)

        weight_w_any = [1./(float(H_any[pres]**exp2*pres**exponent * factor)) for pres in self.presence]
                
        plt.hist2d(self.presence,self.framewise_mean_abs_vel,weights=weight_w_any,**self.hist2d_param)
        plt.xlabel('#pedestrians')
        plt.ylabel('velocity pdf [m/s]')
        plt.title('discrete fundamental diagram')
        
        plt.savefig(PATH_PREFIX + 'discrete_fund_diag.png')


        ####################

        plt.figure()
        
        H_cof = np.bincount(self.presence_cof)

        weight_w_cof = [1./(float(H_cof[pres]**exp2*pres**exponent * factor)) for pres in self.presence_cof]
                
        plt.hist2d(self.presence_cof,self.framewise_mean_abs_vel_cof,weights=weight_w_cof,**self.hist2d_param)
        plt.xlabel('#pedestrians')
        plt.ylabel('velocity pdf [m/s]')
        plt.title('discrete fundamental diagram cof')
        
        plt.savefig(PATH_PREFIX + 'discrete_fund_diag_cof.png')

        #####################

        plt.figure()

        H_ctf = np.bincount(self.presence_ctf)

        weight_w_ctf = [1./(float(H_ctf[pres]**exp2*pres**exponent * factor)) for pres in self.presence_ctf]
                
        plt.hist2d(self.presence_ctf,self.framewise_mean_abs_vel_ctf,weights=weight_w_ctf,**self.hist2d_param)
        plt.xlabel('#pedestrians')
        plt.ylabel('velocity pdf [m/s]')
        plt.title('discrete fundamental diagram ctf')
        
        plt.savefig(PATH_PREFIX + 'discrete_fund_diag_ctf.png')

        ###################

        plt.figure()

        weight_tot = weight_w_any + weight_w_cof + weight_w_ctf
        velocities_tot = self.framewise_mean_abs_vel \
          + self.framewise_mean_abs_vel_cof \
          + self.framewise_mean_abs_vel_ctf

        offset_presence = lambda pr,dx : [v+dx for v in pr]
        presence_tot = offset_presence(self.presence,0) \
          + offset_presence(self.presence_cof,.2)\
          + offset_presence(self.presence_ctf,.4)


        plt.hist2d(presence_tot,velocities_tot,weights=weight_tot,**self.hist2d_param)
        plt.xlabel('#pedestrians')
        plt.ylabel('velocity pdf [m/s]')
        plt.title('discrete fundamental diagram any;cof;ctf')
        
        plt.savefig(PATH_PREFIX + 'discrete_fund_diag_sum.png')

        
        
#        plt.hist2d(self.presence,self.)
    

    def hist_pairwise_distance(self):
        """Output: pairwise distance distributions.
        1,2,3: dx, dy, sqrt(dx^2+dy^2) for all frames with more than one pedestrian;

        4,5,6: dx, dy, sqrt(dx^2+dy^2) for all frames with more than one pedestrian.
               coflow/counterflow/any conditions are compared.
        """
        #complete
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.dist_x,**self.hist_param)
        plt.title('pairwise dx log')
        plt.subplot(1,2,2)
        plt.hist(self.dist_x,**self.hist_param_noLog)
        plt.title('pairwise dx')
        plt.savefig(PATH_PREFIX + 'pairwise_dx.png')

        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.dist_y,**self.hist_param)
        plt.title('pairwise dy log')
        plt.subplot(1,2,2)
        plt.hist(self.dist_y,**self.hist_param_noLog)
        plt.title('pairwise dy')
        plt.savefig(PATH_PREFIX + 'pairwise_dy.png')

        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.dist,**self.hist_param)
        plt.title('pairwise dist log')
        plt.subplot(1,2,2)
        plt.hist(self.dist,**self.hist_param_noLog)
        plt.title('pairwise dist')
        plt.savefig(PATH_PREFIX + 'pairwise_dist.png')


        #coflow vs. counterflow

        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.dist_x_cof,**self.hist_param)
        plt.hist(self.dist_x_ctf,**self.hist_param)
        plt.hist(self.dist_x,**self.hist_param)
        plt.legend(('cof','ctf','all'))
        plt.title('pairwise dx log')
        plt.subplot(1,2,2)
        plt.hist(self.dist_x_cof,**self.hist_param_noLog)
        plt.hist(self.dist_x_ctf,**self.hist_param_noLog)
        plt.hist(self.dist_x,**self.hist_param_noLog)
        plt.title('pairwise dx')
        plt.savefig(PATH_PREFIX + 'pairwise_cofctf_dx.png')

        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.dist_y_cof,**self.hist_param)
        plt.hist(self.dist_y_ctf,**self.hist_param)
        plt.hist(self.dist_y,**self.hist_param)
        plt.legend(('cof','ctf','all'))
        plt.title('pairwise dy log')
        plt.subplot(1,2,2)
        plt.hist(self.dist_y_cof,**self.hist_param_noLog)
        plt.hist(self.dist_y_ctf,**self.hist_param_noLog)
        plt.hist(self.dist_y,**self.hist_param_noLog)
        plt.title('pairwise dy')
        plt.savefig(PATH_PREFIX + 'pairwise_cofctf_dy.png')

        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.dist_cof,**self.hist_param)
        plt.hist(self.dist_ctf,**self.hist_param)
        plt.hist(self.dist,**self.hist_param)
        plt.legend(('cof','ctf','all'))
        plt.title('pairwise dist')
        plt.subplot(1,2,2)
        plt.hist(self.dist_cof,**self.hist_param_noLog)
        plt.hist(self.dist_ctf,**self.hist_param_noLog)
        plt.hist(self.dist,**self.hist_param_noLog)
        plt.title('pairwise dist')
        plt.savefig(PATH_PREFIX + 'pairwise_cofctf_dist.png')

    def hist_distance_velocity(self,\
                                   power=1,\
                                   log=False,\
                                   xmax=2.5,\
                                   vmin=0,\
                                   vmax=2,\
                                   normmode='fixed'):
        """Output: Histogram of the joint distribution of dX vs. normalized dV.
        Input: power=1,\
               log=False,\
               xmax=2.5,\
               vmin=0,\
               vmax=2,\
               normmode='fixed' | 'casedependent'
        """
        cRMS = lambda dv : np.sqrt(np.mean(np.array(dv)**2))
        self.delta_v_rms = cRMS(self.delta_v)
        self.delta_v_cof_rms = cRMS(self.delta_v_cof)
        self.delta_v_ctf_rms = cRMS(self.delta_v_ctf)


        hist2d_param_local = copy.deepcopy(self.hist2d_param)        
        power_str = "%3g" % power
 
        
        
        if log:
            hist2d_param_local['norm'] = LogNorm()
            log_str = '_log' 
        else:
            log_str = ''
        
        feat_str = power_str + log_str


        hist2d_param_local['range']= [[0,xmax],[vmin,vmax]]
        
        genNormalizedPower = lambda dv_vector,norm,p : [ (dv/norm)**p for dv in dv_vector ]

        if normmode=='fixed':
            normvalues = {'any':self.delta_v_rms , 'cof':self.delta_v_rms, 'ctf':self.delta_v_rms}
            normvalues_string = {'any':'' , 'cof':'', 'ctf':''}
        elif normode=='casedependent':
            normvalues = {'any':self.delta_v_rms , 'cof':self.delta_v_cof_rms, 'ctf':self.delta_v_ctf_rms}
            normvalues_string = {'any':'' , 'cof':'_cof', 'ctf':'_ctf'}

        
        plt.figure()
        plt.hist2d(self.dist,genNormalizedPower(self.delta_v,normvalues['any'],power),**hist2d_param_local)
        plt.xlabel('dX [m]')
        plt.ylabel('(|dV|/RMS(dV))^p')
        plt.title('|dX| vs. (|dV|/RMS(dV%s))^p, RMS(dV%s)=%5g, p=%s' % (\
                normvalues_string['any'],normvalues_string['any'],normvalues['any'], feat_str))
        plt.colorbar()
        plt.savefig(PATH_PREFIX + 'dx_vs_dv_p=' + feat_str + '.png')

        plt.figure()
        plt.hist2d(self.dist_cof,genNormalizedPower(self.delta_v_cof,normvalues['cof'],power),**hist2d_param_local)
        plt.xlabel('dX [m]')
        plt.ylabel('(|dV|/RMS(dV))^p')
        plt.title('|dX| vs. (|dV|/RMS(dV%s))^p, cof, RMS(dV%s)=%g, p=%s' %(\
                normvalues_string['cof'],normvalues_string['cof'],normvalues['cof'],feat_str))
        plt.colorbar()
        plt.savefig(PATH_PREFIX + 'dx_vs_dv_cof_p='+ feat_str +'.png')

        plt.figure()
        plt.hist2d(self.dist_ctf,genNormalizedPower(self.delta_v_ctf,normvalues['ctf'],power),**hist2d_param_local)
        plt.xlabel('dX [m]')
        plt.ylabel('(|dV|/RMS(dV))^p')
        plt.title('|dX| vs. (|dV|/RMS(dV%s))^p, ctf, RMS(dV%s)=%5g, p=%s' % (
                normvalues_string['ctf'],normvalues_string['ctf'],normvalues['ctf'],feat_str))
        plt.colorbar()
        plt.savefig(PATH_PREFIX + 'dx_vs_dv_ctf_p=' + feat_str + '.png')

                
    
    
    def hist_absolute_direction_distribution(self):
        
        direction_distribution = np.array([len(t) for t in [self.going_to_the_right, self.going_to_the_left]])
    
        plt.figure()
        plt.bar([-1, 1] , direction_distribution)
        plt.title('absolute direction distrib (2R;2L)')
        plt.savefig(PATH_PREFIX + 'absolute_dir_distrib.png')

    def hist_presence_distribution(self):
        plt.figure()
        plt.subplot(121)
        plt.hist(self.presence,**self.hist_param)
        plt.title('absolute presence distribution log')

        plt.subplot(122)
        plt.hist(self.presence, histtype='step',normed=True)
        plt.title('absolute presence distribution')
        plt.savefig(PATH_PREFIX + 'absolute_presence_distribution_all.png')

        
        plt.figure()
        plt.subplot(121)
        plt.hist(self.presence_cof,**self.hist_param_noNormed)
        plt.hist(self.presence_ctf,**self.hist_param_noNormed)
        plt.hist(self.presence,**self.hist_param_noNormed)
        plt.legend(('cof','ctf','all'))
        plt.title('absolute presence distribution log')

        plt.subplot(122)
        plt.hist(self.presence_cof,alpha=.5, histtype='step')
        plt.hist(self.presence_ctf,alpha=.5, histtype='step')
        plt.hist(self.presence,alpha=.5, histtype='step')
        plt.title('absolute presence distribution')
        plt.savefig(PATH_PREFIX + 'absolute_presence_distribution.png')

    def hist_position_pdf(self):        
        plt.figure()
        plt.hist2d(self.single_x,self.single_y,**self.hist2d_param)
        plt.colorbar()
        plt.title('pdf of single pedestrian position')
        plt.savefig(PATH_PREFIX + 'single_pdf_pos.png')

        plt.figure()
        plt.hist2d(self.single_x_gt,self.single_y_gt,**self.hist2d_param)
        plt.colorbar()
        plt.title('pdf of single pedestrian just 2R')
        plt.savefig(PATH_PREFIX + 'single_pdf_pos2R.png')

        plt.figure()
        plt.hist2d(self.single_x_lt,self.single_y_lt,**self.hist2d_param)
        plt.colorbar()
        plt.title('pdf of single pedestrian position just 2L')
        plt.savefig(PATH_PREFIX + 'single_pdf_pos2L.png')

        plt.figure()
        plt.hist2d(self.tot_x,self.tot_y,**self.hist2d_param)
        plt.colorbar()
        plt.title('pdf of pedestrian position')
        plt.savefig(PATH_PREFIX + 'multi_pdf_pos.png')

    @staticmethod
    def normfit_over_bins(data,bins,color,data_complete=None):
        (mu,sigma) = norm.fit(data)

        #(k2,pv) = normaltest(data)
        Y = mlab.normpdf(bins,mu,sigma)

        if data_complete != None:
            ratio = (1.*len(data))/(1.* len(data_complete))
        else:
            ratio = 1

        Y = ratio*mlab.normpdf(bins,mu,sigma)
        
        
        plt.plot(bins,Y,color + '--')

        return {'mu': mu,'sigma':sigma}#,'k2':k2,'pv':pv}

        
    
    def pltalias_hist_and_smooth(self,data,color,smoothed=False,sigma = 1.5,hist_alpha=1,normFit=True,ylim=1e-5):
        temp_alpha = self.hist_param['alpha']
        self.hist_param['alpha'] = hist_alpha
        n,bins,patches = plt.hist(data,color=color,**self.hist_param)
        if smoothed:
            smooth_hist(n,bins,color,sigma)        
        
        self.hist_param['alpha'] = temp_alpha

        ret = {'n':n,'bins': bins,'patches':patches}
        
        if normFit:
            ret_dict = self.normfit_over_bins(data,bins,color)
            plt.ylim(ymin=ylim)

            return dict(ret_dict.items() + ret.items())

        else:
            return ret 

    def hist_single_ped_stats(self,**kw):#sigma = 1.5):

        
        plt.figure()
        #plt.hist(self.single_u_gt,color='r',bins=80,normed=True,log=True,alpha=.5)
        #plt.hist(self.single_u_gt,color='r',**self.hist_param)
        #plt.hist(self.single_u_lt,color='b',**self.hist_param)

        self.pltalias_hist_and_smooth(self.single_u_gt,color='r',**kw)        
        self.pltalias_hist_and_smooth(self.single_u_lt,color='b',**kw)

        
        
        plt.legend(('2R','2L'))
        plt.xlabel('vx [m/s]')
        plt.title('vx distribution, single ped')
        plt.savefig(PATH_PREFIX + 'single vx.png')


        plt.figure()
        #plt.hist(self.single_v_gt,color='r')
        #plt.hist(self.single_v_lt,color='b')

        self.pltalias_hist_and_smooth(self.single_v_gt,color='r',**kw)
        self.pltalias_hist_and_smooth(self.single_v_lt,color='b',**kw)

        

        plt.legend(('2R','2L'))
        plt.xlabel('vy [m/s]')
        plt.title('vy distribution, single ped')
        plt.savefig(PATH_PREFIX +'single vy.png')


        plt.figure()
        #plt.hist(self.single_ax_gt,color='r',**self.hist_param)
        #plt.hist(self.single_ax_lt,color='b',**self.hist_param)

        self.pltalias_hist_and_smooth(self.single_ax_gt,color='r',**kw)
        self.pltalias_hist_and_smooth(self.single_ax_lt,color='b',**kw)

        
        
        plt.legend(('2R','2L'))
        plt.xlabel('ax [m/s2]')
        plt.title('ax distribution, single ped')
        plt.savefig(PATH_PREFIX +'single ax.png')


        plt.figure()
        #plt.hist(self.single_ay_gt,color='r',**self.hist_param)
        #plt.hist(self.single_ay_lt,color='b',**self.hist_param)

        self.pltalias_hist_and_smooth(self.single_ay_gt,color='r',**kw)
        self.pltalias_hist_and_smooth(self.single_ay_lt,color='b',**kw)

        

        plt.legend(('2R','2L'))
        plt.xlabel('ay [m/s2]')
        plt.title('ay distribution, single ped')
        plt.savefig(PATH_PREFIX +'single ay.png')
    

    def hist_multi_ped_stats(self,u_threshold=.2,**kw):#,sigma = 1.5):

        # av_bins = lambda bins : .5*(bins[1:]+bins[:-1])
        # sm_n = lambda n : filters.gaussian_filter(n,sigma)
        # smooth_hist = lambda n,bins,col : plt.plot(av_bins(bins),sm_n(n),color=col)

        # def hist_and_smooth_alias(data,color):
        #     n,bins,patches = plt.hist(data,color=color,**self.hist_param)
        #     smooth_hist(n,bins,color)        
        

        plt.figure()
        # n,bins,patches = plt.hist(self.multi_u_cof,color='r',**self.hist_param)
        # smooth_hist(n,bins,'r')        
        
        # n,bins,patches = plt.hist(self.multi_u_ctf,color='b',**self.hist_param)
        # smooth_hist(n,bins,'b')

        # n,bins,patches = plt.hist(self.single_u,color='g',**self.hist_param)
        # smooth_hist(n,bins,'g') 

        
        
        ret = self.pltalias_hist_and_smooth(self.multi_u_cof,color='r',normFit=False,**kw)

        
        
        self.normfit_over_bins([u for u in self.multi_u_cof if u > u_threshold ],ret['bins'],'r',self.multi_u_cof)
        self.normfit_over_bins([u for u in self.multi_u_cof if u < -u_threshold ],ret['bins'],'r',self.multi_u_cof)
        
        ret = self.pltalias_hist_and_smooth(self.multi_u_ctf,color='b',normFit=False,**kw)
        self.normfit_over_bins([u for u in self.multi_u_ctf if u > u_threshold ],ret['bins'],'b',self.multi_u_ctf)
        self.normfit_over_bins([u for u in self.multi_u_ctf if u < -u_threshold ],ret['bins'],'b',self.multi_u_ctf)
        
        ret = self.pltalias_hist_and_smooth(self.single_u,color='g',normFit=False,**kw)

        self.normfit_over_bins([u for u in self.single_u if u > u_threshold ],ret['bins'],'g',self.single_u)
        self.normfit_over_bins([u for u in self.single_u if u < -u_threshold ],ret['bins'],'g',self.single_u)

        plt.ylim(ymin=1e-5)       


        plt.legend(('cof','cof','ctf','ctf','sin','sin'))
        plt.xlabel('vx [m/s]')
        plt.title('vx distribution, multi ped')
        plt.savefig(PATH_PREFIX +'multi vx.png')

        plt.figure()
        # n,bins,patches = plt.hist(self.multi_v_cof,color='r',**self.hist_param)
        # smooth_hist(n,bins,'r')
        
        # n,bins,patches = plt.hist(self.multi_v_ctf,color='b',**self.hist_param)
        # smooth_hist(n,bins,'b')

        # n,bins,patches = plt.hist(self.single_v,color='g',**self.hist_param)
        # smooth_hist(n,bins,'g')

        self.pltalias_hist_and_smooth(self.multi_v_cof,color='r',**kw)
        self.pltalias_hist_and_smooth(self.multi_v_ctf,color='b',**kw)
        self.pltalias_hist_and_smooth(self.single_v,color='g',**kw)                            



        plt.legend(('cof','ctf','sin'))
        plt.xlabel('vy [m/s]')
        plt.title('vy distribution, multi ped')
        plt.savefig(PATH_PREFIX +'multi vy.png')

        plt.figure()
        # plt.hist(self.multi_ax_cof,color='r',**self.hist_param)
        # plt.hist(self.multi_ax_ctf,color='b',**self.hist_param)
        # plt.hist(self.single_ax,color='g',**self.hist_param)

        self.pltalias_hist_and_smooth(self.multi_ax_cof,color='r',**kw)
        self.pltalias_hist_and_smooth(self.multi_ax_ctf,color='b',**kw)
        self.pltalias_hist_and_smooth(self.single_ax,color='g',**kw)


        plt.legend(('cof','ctf','sin'))
        plt.xlabel('ax [m/s2]')
        plt.title('ax distribution, multi ped')
        plt.savefig(PATH_PREFIX +'multi ax.png')


        plt.figure()
        # plt.hist(self.multi_ay_cof,color='r',**self.hist_param)
        # plt.hist(self.multi_ay_ctf,color='b',**self.hist_param)
        # plt.hist(self.single_ay,color='g',**self.hist_param)

        self.pltalias_hist_and_smooth(self.multi_ay_cof,color='r',**kw)
        self.pltalias_hist_and_smooth(self.multi_ay_ctf,color='b',**kw)
        self.pltalias_hist_and_smooth(self.single_ay,color='g',**kw)

        plt.legend(('cof','ctf','sin'))

        plt.xlabel('ay [m/s2]')
        plt.title('ay distribution, multi ped')
        plt.savefig(PATH_PREFIX +'multi ay.png')


        plt.figure()

        self.pltalias_hist_and_smooth(self.multi_speed_cof,color='r',normFit=False,**kw)
        self.pltalias_hist_and_smooth(self.multi_speed_ctf,color='b',normFit=False,**kw)
        self.pltalias_hist_and_smooth(self.single_speed,color='g',normFit=False,**kw)

        plt.legend(('cof','ctf','sin'))

        plt.xlabel('speed [m/s]')
        plt.title('speed, multi ped')
        plt.savefig(PATH_PREFIX +'multi_speed.png')

        plt.figure()

        self.pltalias_hist_and_smooth(self.multi_acc_cof,color='r',normFit=False,**kw)
        self.pltalias_hist_and_smooth(self.multi_acc_ctf,color='b',normFit=False,**kw)
        self.pltalias_hist_and_smooth(self.single_acc,color='g',normFit=False,**kw)

        plt.legend(('cof','ctf','sin'))

        plt.xlabel('acceleration [m/s2]')
        plt.title('acceleration, multi ped')
        plt.savefig(PATH_PREFIX +'multi_speed.png')


    def hist_custom_qty(self,qtys,colors,**kw):
        plt.figure()
        if colors == None:
            pass
            #for qty in qtys:
            #    self.pltalias_hist_and_smooth(qty,**kw)

        else:
            for qty,col in zip(qtys,colors):
                self.pltalias_hist_and_smooth(qty,color=col,**kw)




            



######### PLOTTING

def plot_traj(traj,fig=None):
    """ plots single trajectory as dots, curves and arrows
    using matplotlib
    """
    if fig == None:
        plt.figure()
        
    x,y,z,u,v,w = [],[],[],[],[],[]
    for p in traj: # for all particles
        x.append(p.x)
        y.append(p.y)
        z.append(p.z)
        u.append(p.u)
        v.append(p.v)
        w.append(p.w)
        
    plt.plot(x,y,'o--')
    #plt.quiver(x,y,u,v)
    plt.axis('equal')
        
 
def plot_all_trajectories(list_of_traj):
    """ plots all the trajectories in a given list of trajectories on a single
    figure, overlapping the curves
    """
    fig = plt.figure()
    plt.hold(True)
    for traj in list_of_traj:
           plot_traj(traj,fig)
    #fig.show(block=False)


def plot_colored_trajectories(list_of_traj):
    """ plots all the trajectories in a given list of trajectories on a single
    figure, overlapping the curves
    """
    fig = plt.figure()
    plt.hold(True)
    for traj in list_of_traj:
        x,y,z,u,v,w = [],[],[],[],[],[]
        for p in traj: # for all particles
            x.append(p.x)
            y.append(p.y)
            z.append(p.z)
            u.append(p.u)
            v.append(p.v)
            w.append(p.w)
        
        if np.median(u) > 0:
            plt.plot(x[0],y[0],'rs',markersize=10)
            plt.plot(x,y,'bo--')
            plt.quiver(x,y,u,v,color='b')
        else:
            plt.plot(x[0],y[0],'bo',markersize=10)
            plt.plot(x,y,'rs-.')
            plt.quiver(x,y,u,v,color='r')   
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Direction-colored velocity map')
    plt.axis('equal')
    #fig.show()
 
