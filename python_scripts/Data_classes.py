import numpy as np



def sentence(sentence):
    print sentence
def sentence_done():
    print "\tdone!"



DIRECTORY_SET_FNAME = 'directory_set.list'
def openDirectorySet(Tag = None):
    # with open(DIRECTORY_SET_FNAME) as f:
    #     content = f.readlines()

    content = [line.rstrip('\n') for line in open(DIRECTORY_SET_FNAME)]
    content = [ line for line in content if line[0]!= '#']

    

    return content
    

DATA_F_NAME = 'ptv_is.npz'    
def openDataSetFromDir(directory):

    print "processing directory %s" % directory
    
    data_directory = '../' + directory + '/output_trk/results/'
    ####

    sentence("loading datafile...")
    data_file = np.load(data_directory + DATA_F_NAME)

    data = data_file['data']
    traj = data_file['traj']

    sentence_done()

    return (data,traj)





class Particle(object):
    """ Particle object that defines every single tracked point
    the attributes are position (x,y,z) and velocities (u,v,w), time instance (t)
    and trajectory identity (id)
    """
    UNIDENTIFIED_PARTICLE=-99
    
    
    def is_unidentified(self):
        """ Checks wheter the particle has a proper id or not.
        Returns true if self.id==UNIDENTIFIED_PARTICLE(=99)
        """
        return self.id==Particle.UNIDENTIFIED_PARTICLE
    
    def __init__(self,p=0,n=0, x=0.0,y=0.0,z=0.0,t=0,id=UNIDENTIFIED_PARTICLE,\
    u=0.0,v=0.0,w=0.0,ax=0.0,ay=0.0,az=0.0):
        self.p = p
        self.n = n
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.id = id
        self.u = u
        self.v = v
        self.w = w
        self.ax = ax
        self.ay = ay
        self.az = az
        
    def __str__(self):
        return '%.2f %.2f %.2f %.2f %.2f %.2f %d %d' % (self.x,self.y,self.z,\
        self.u,self.v,self.w,self.t,self.id)
        
    def __repr__(self):
        return str(self)


class Detection:
    UNIDENTIFIED_PARTICLE=-99
    def __init__(self
                 , x = 0
                 , y = 0
                 , z = 0.
                 , ell = False
                 , cx = 0.
                 , cy = 0.
                 , A = 0.
                 , B = 0.
                 , C = 0.
                 , id = UNIDENTIFIED_PARTICLE ):\
                 
        self.x = x
        self.y = y
        self.z = z
        self.ellipse_capt = ell
        self.cx = cx
        self.cy = cy
        self.A = A
        self.B = B
        self.C = C
        self.id = id

    def __str__(self):
        # return '%.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (self.x
        #                                           , self.y
        #                                           , self.cx
        #                                           , self.cy
        #                                           , self.A
        #                                           , self.B
        #                                           , self.C)
        return 'H=(%d %d %.2f) E=(%.2f %.2f %.2f %.2f %.2f)'  % (self.x
                                                  , self.y
                                                  , self.z
                                                  , self.cx
                                                  , self.cy
                                                  , self.A
                                                  , self.B
                                                  , self.C)
#                                                  , self.id)

    def __repr__(self):
        return str(self)


        
class Frame(list):
    """ Frame class is a list of Particles in the given frame, sharing the same
    time instance (t)
    """
    def __new__(cls, data=None,len=0):
        obj = super(Frame, cls).__new__(cls, data)
        return obj

    def __str__(self):
        s = 'Frame(['
        for i in self:
            s += str(i) + ' | '
        s += '])'
        return s
        # return 'Traj(%s)' % list(self)
        
    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Frame(list(self) + list(other))
        
    def len(self):
        return len(list(self))

    #def getTimeInstant(self):
    #    return 
        

class Trajectory(list):
    """ Trajectory class is a list of Particles that are linked together in 
    time
    """
    def __new__(cls, data=None):
        obj = super(Trajectory, cls).__new__(cls, data)
        return obj

    def __str__(self):
        s = 'Traj(['
        for i in self:
            s += str(i) + ' | '
        s += '])'
        return s
        # return 'Traj(%s)' % list(self)
        
    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Trajectory(list(self) + list(other))

    


class Id2traj_idx:
    """To fix the possibile problem:
    traj[num].id != num
    """
    def __getitem__(self,item):
        return self.id2traj_[item]
    
    def __init__(self, traj):
        max_id = 0
        for t in traj:
            max_id = max(max_id,t[0].id)

        #max_id = np.max(np.asarray([t[0].id for t in traj]))
        
        self.id2traj_ = [-1] * (max_id+1)
        for idx,t in enumerate(traj):
            #print t[0].id,max_id
            self.id2traj_[t[0].id] = idx
             
