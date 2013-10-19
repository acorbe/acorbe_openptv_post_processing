import pandas as pd
import numpy as np
import traj_processing as tp
import matplotlib.pyplot as plt
import itertools as it 
from scipy.stats import maxwell

PATH_PREFIX = '/home/acorbe/datas/'

def get_qty_from_frame(f,getter):
    return [getter(p) for p in f]

hist_param = {'bins':80 , 'normed':True, 'alpha':.5, 'histtype':'step', 'log':True}
hist_param_noLog = {'bins':80 , 'normed':True, 'alpha':.5, 'histtype':'step' }

hist2d_param = {'bins':80,'normed':True}



in_frame_selector_u_median = lambda f,comp : [[ comp(u,0) for u in f['u_median']  ]]
in_frame_selector_u_median_lt = \
    lambda f : in_frame_selector_u_median(f, lambda x,y : x<y )
in_frame_selector_u_median_gt = \
    lambda f : in_frame_selector_u_median(f, lambda x,y : x>y )



def elaborate_univariate_stats(crStats
                               , getter
                               , in_frame_selector=None
                               , further_condtioning=None
                               , hist_output=True
                               , normalization=1.
                               , fit_with_normal = False
                               , col = 'b'
                               , **kw
                               ):

    if further_condtioning is None:
        sub_set_data = crStats.datas['data']
        sub_set_all = crStats.datas
    else:
        sub_set_data = crStats.datas[further_condtioning]['data']
        sub_set_all = crStats.datas[further_condtioning]

    unflattened_output = sub_set_data.apply( lambda f : get_qty_from_frame( f,getter ) )
    
    if in_frame_selector is not None:
        in_frame_selector_data = sub_set_all.apply(in_frame_selector,axis=1)['data']
        #in_frame_selector_data['unfl_output'] = unflattened_output
        #in_frame_selector_data.apply( lambda f : [ val['unfl_output'] for val in f if val['data'] ] )['unfl_output']

        
        
        #output = [val for f in in_frame_selector for val in f]
        #sub_selection
        output = [val/normalization for f,s in it.izip(unflattened_output,in_frame_selector_data) for val,cond in zip(f,s) if cond]
    else:
        output = [val/normalization for f in unflattened_output for val in f]

    output = np.asarray(output)
    unflattened_output = np.asarray(unflattened_output)

    #print kw.values()
    final_hist_param = dict( hist_param.items() + kw.items() )
   
    bins = None
    
    if hist_output:
        output_t = np.asarray(output)
        output_t = output_t[~np.isnan(output)]
        ax = plt.gca()
        next_color = next (ax._get_lines.color_cycle)
        n,bins,pathces = plt.hist(output_t,color=next_color,**final_hist_param)
        

        if fit_with_normal:
            tp.LagrangianStats.normfit_over_bins(output_t,bins,next_color + '.')
            
            nmin_ =  np.min(n[n>0])
            
            plt.ylim(ymin = nmin_)
        plt.grid('on')

    return output,bins,unflattened_output

def elaborate_bivariate_stats(crStats
                              , getter1
                              , getter2
                              , further_conditioning=None
                              , hist_output=True
                              , **kw
                              ):

    out1,bins1,u_out1 = elaborate_univariate_stats(crStats
                                             ,getter1
                                             ,further_conditioning
                                             ,hist_output=False)

    
    out2,bins2,u_out2 = elaborate_univariate_stats(crStats
                                             ,getter2
                                             ,further_conditioning
                                             ,hist_output=False)



    final_hist_param = dict( hist2d_param.items() + kw.items() )
   
    
    if hist_output:
        plt.hist2d(out1,out2,**final_hist_param)

    return out1,out2,u_out1,u_out2



def elaborate_single_ped_stats(crStats):

    sin_ped_idx = crStats.datas['fmeasure'] == 1
    sin_ped = crStats.datas[sin_ped_idx]['data']


    getter = tp.get_u
    sin_ped_unfl = sin_ped.apply( lambda f : get_qty_from_frame( f,getter ) )

    sin_ped = [val for f in sin_ped_unfl for val in f]

    plt.hist(sin_ped,bins=80)
    


def sub_eval_desired_velocity_qtys(crStats
                                ,further_conditioning = None                
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


        x_val,t1,t2 = elaborate_univariate_stats(crStats, tp.get_x, further_conditioning = further_conditioning , hist_output = False)

        y_val,t1,t2 = elaborate_univariate_stats(crStats, tp.get_y, further_conditioning = further_conditioning , hist_output = False)

        u_val,t1,t2 = elaborate_univariate_stats(crStats, tp.get_u, further_conditioning = further_conditioning , hist_output = False)

        v_val,t1,t2 = elaborate_univariate_stats(crStats, tp.get_v, further_conditioning = further_conditioning , hist_output = False)

        remove_if_nan = lambda li : [e for e,v in zip(li,v_val) if not np.isnan(v) ] 

        x_val = remove_if_nan(x_val)
        y_val = remove_if_nan(y_val)
        u_val = remove_if_nan(u_val)
        v_val = remove_if_nan(v_val)
        
        #x_val = self.all_stats['single']['x'][spec_type]
        #y_val = self.all_stats['single']['y'][spec_type]
        #u_val = self.all_stats['single'][field_x][spec_type]
        #v_val = self.all_stats['single'][field_y][spec_type]
        
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

def quiver_desired_velocity(crStats\
                                ,further_conditioning = None
                                ,grid_x=40,grid_y=40,small_incr=.00001\
                                ,qty='vel'
                                ,spec_type='gt'
                                ,img_format='png'
                                ,**kw):


        x_class_v,y_class_v,u_means,v_means,u_delta,v_delta,u_all,v_all =\
            sub_eval_desired_velocity_qtys(crStats
                                        ,further_conditioning\
                                               ,grid_x,grid_y,small_incr\
                                               ,qty\
                                               ,spec_type)


        plt.figure()
        plt.hist2d(u_delta,v_delta,**hist2d_param)
        plt.title('Overall noise distribution')
        plt.savefig(PATH_PREFIX + 'total_noise_%s_%s.%s' % (qty,spec_type,img_format))


        rv = maxwell()

        speed_delta = np.sqrt(np.asarray(u_delta)**2 + np.asarray(v_delta)**2)

        loc,scale = maxwell.fit(speed_delta)
        
        plt.figure()
        plt.hist(speed_delta,**hist_param_noLog)
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

# def hist_desired_speed_temperature(self\
#                                 ,grid_x=40,grid_y=40,small_incr=.00001\
#                                 ,qty='vel'
#                                 ,img_format='png'):
                                

    
#         x_class_v_g,y_class_v_g,u_means_g,v_means_g,u_delta_g,v_delta_g,u_all_g,v_all_g =\
#            self.sub_eval_desired_velocity_qtys(\
#                                                grid_x,grid_y,small_incr\
#                                                ,qty\
#                                                ,spec_type='gt')
      

#         speed_delta_g = np.sqrt(np.asarray(u_delta_g)**2 + np.asarray(v_delta_g)**2)


#         x_class_v_g,y_class_v_g,u_means_g,v_means_g,u_delta_g,v_delta_g,u_all_g,v_all_g =\
#            self.sub_eval_desired_velocity_qtys(\
#                                                grid_x,grid_y,small_incr\
#                                                ,qty\
#                                                ,spec_type='lt')

       

#         speed_delta_l = np.sqrt(np.asarray(u_delta_g)**2 + np.asarray(v_delta_g)**2)

#         plt.figure()
        
#         plt.hist(speed_delta_g,**self.hist_param_noLog)
#         plt.hist(speed_delta_l,**self.hist_param_noLog)
        
#         plt.title('Speed fluctuation comparison')
#         plt.legend(['2R','2L'])

#         plt.savefig(PATH_PREFIX + 'speed_delta_comp_%s.%s' % (qty,img_format))

    
