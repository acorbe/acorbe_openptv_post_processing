
import corr_process as cp_
import numpy as np

if __name__ == '__main__':
    std_ = np.linspace(.01,3,20)

    a_rms = []
    a_max = []
    a_rms_spline = []
    a_max_spline = []

    
    ax_rms = []
    ay_rms = []
    ax_rms_spline = []
    ay_rms_spline = []
    

    ax_max = []
    ay_max = []
    ax_max_spline = []
    ay_max_spline = []

        

    for idx,std in enumerate(std_):

        print ' '
        print ' '
        print ' '
        print ' '
        print "process %d of %d" % ( idx , len(std_) )
        

        
        crStats = cp_.main( useDetectionDb = False )
        
        crStats.apply_gaussian_filter_on_tracks(std)
        crStats.compute_derivatives_spline()
        crStats.calculate_trajwise_stats()
        
        a_rms_spline.append( [ crStats.trajs['a_rms'].values.copy() ] )
        a_max_spline.append( [ crStats.trajs['a_max'].values.copy() ] )
        ax_rms_spline.append( [ crStats.trajs['ax_rms'].values.copy() ] )
        ay_rms_spline.append( [ crStats.trajs['ay_rms'].values.copy() ] )
        ax_max_spline.append( [ crStats.trajs['ax_max'].values.copy() ] )
        ay_max_spline.append( [ crStats.trajs['ay_max'].values.copy() ] )

                
        crStats.compute_derivatives_fd()
        crStats.calculate_trajwise_stats()
        a_rms.append( [ crStats.trajs['a_rms'].values.copy() ] )
        a_max.append( [ crStats.trajs['a_max'].values.copy() ] )

        ax_rms.append( [ crStats.trajs['ax_rms'].values.copy() ] )
        ay_rms.append( [ crStats.trajs['ay_rms'].values.copy() ] )
        ax_max.append( [ crStats.trajs['ax_max'].values.copy() ] )
        ay_max.append( [ crStats.trajs['ay_max'].values.copy() ] )

        






        



        

        
        
        


