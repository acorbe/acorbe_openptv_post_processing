
import traj_correlation as cr
import Data_classes as dtCl
import tracks_showoff as ts
import traj_processing as tp
import ensemble_statistics as es
import pandas as pd
import analyze_framerate as af
import select_good_trajectories as sgt
import par_estimation as pe


def main(useDetectionDb = False ):

    crStats = cr.Correlations()
    
    directory_set = dtCl.openDirectorySet()
    for directory in directory_set:        
       (data,traj) = dtCl.openDataSetFromDir(directory)
       crStats.acquire_data_sets(data)#,traj)

    
    #crStats.recalculate_derivatives_spline()
    crStats.finalize_acquisition()    
    crStats.calculate_trajwise_stats()
    crStats.build_selection_criteria()
    #crStats.single_ped_structure_function()
    
    is_tracks_compl = crStats.criteria['isolate_tracks_complete']
    is_tracks_lt_N = crStats.criteria['isolate_tracks_shorter_than_N']
    is_tracks_gt_N = crStats.criteria['isolate_tracks_longer_than_N']

    detection_DB = None

    if useDetectionDb:

        print "loading detection_DB..."
        detection_DB_FNAME = 'detection_DB.h5'
        store = pd.HDFStore(detection_DB_FNAME)
        detection_DB = store['detection_DB']
        store.close()
        print "done"

        print "resorting detection_DB..."
        detection_DB = detection_DB.sortlevel(0)    
        print "done!"

        print "attaching correct timestamps..."
        af.attach_correct_time_stamp(detection_DB , crStats)
        print "done!"

        print "analyzing framerate..."
        crStats.analyse_framerate()
        print "done!"

        sgt.define_criteria_good_tracks_default(crStats)

        

    print ""
    print ""
    print  "READY!"

    return crStats,detection_DB


if __name__ == '__main__':

    crStats,detection_DB = main(useDetectionDb = True)
    
