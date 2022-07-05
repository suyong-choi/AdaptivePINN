import numpy as np

class GTree:
    type_AND = 0
    type_OR = 1
    type_NOT = 2
    type_IDENTITY = 3

    ppstring = [' AND ', ' OR ', ' NOT ']

    def __init__(self, obj=None):
        self.tree = None
        self.type = GTree.type_IDENTITY
        self.daughters = [obj]
        pass

    def add(self, type, geomlist):
        self.type = type
        self.daughters = geomlist
        pass

    # build mini tree out of simple operations
    def AND(geom1, geom2):
        gt = GTree()
        gt.add(GTree.type_AND, [geom1, geom2])
        return gt

    def OR(geom1, geom2):
        gt = GTree()
        gt.add(GTree.type_OR, [geom1, geom2])
        return gt
    
    def NOT(geom1):
        gt = GTree()
        gt.add(GTree.type_NOT, [geom1])
        return gt

    # to print the object
    def __str__(self):
        res = ''
        if self.type == GTree.type_AND or self.type == GTree.type_OR:
            res = '(' + self.daughters[0].__str__() + GTree.ppstring[self.type] + self.daughters[1].__str__() + ')'
        elif self.type == GTree.type_NOT:
            res = ' (~' + self.daughters[0].__str__() + ') '
        elif self.type == GTree.type_IDENTITY:
            res = self.daughters[0].__str__()
        return res

    def is_inside(self, coords):
        res = False
        if self.type == GTree.type_AND:
            res = np.logical_and(self.daughters[0].is_inside(coords), self.daughters[1].is_inside(coords))
        elif self.type == GTree.type_OR:
            res = np.logical_or(self.daughters[0].is_inside(coords), self.daughters[1].is_inside(coords))
        elif self.type == GTree.type_NOT:
            res = np.logical_not(self.daughters[0].is_inside(coords))
        elif self.type ==GTree.type_IDENTITY:
            res = self.daughters[0].is_inside(coords)
        return res

    def generate_points(self, npoints):
        trialpoints = self.daughters[0].generate_points(npoints, interior=True)
        sel = self.is_inside(trialpoints)
        selected = trialpoints[sel]
        return selected



class geom:
    def __init__(self):
        self.geom_tree = []
        pass

    def generate_points(self, npoints, interior=False):
        pass

    def is_inside(self, coords):
        pass


    
class geom_rect(geom):
    def __init__(self, xcenter, ycenter, width, height):
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.width = width
        self.height = height

    def generate_points(self, npoints, interior=False):
        if interior:
            x = np.random.uniform(self.xcenter - 0.5*self.width, self.xcenter + 0.5*self.width, size=(npoints, 1)).astype(np.float32)
            y = np.random.uniform(self.ycenter - 0.5*self.height, self.ycenter + 0.5*self.height, size=(npoints, 1)).astype(np.float32)
            all_coords = np.hstack((x, y))
        else:
            ypwall = np.random.uniform(self.xcenter - 0.5*self.width, self.xcenter + 0.5*self.width, size=(npoints, 1)).astype(np.float32)
            ymwall = np.random.uniform(self.xcenter - 0.5*self.width, self.xcenter + 0.5*self.width, size=(npoints, 1)).astype(np.float32)

            xpwall = np.random.uniform(self.ycenter - 0.5*self.height, self.ycenter + 0.5*self.height, size=(npoints, 1)).astype(np.float32)
            xmwall = np.random.uniform(self.ycenter - 0.5*self.height, self.ycenter + 0.5*self.height, size=(npoints, 1)).astype(np.float32)

            
            xpwall_coords = np.hstack( (np.array([self.xcenter+0.5*self.width]*npoints, dtype=np.float32).reshape((npoints, 1)), xpwall) )
            xmwall_coords = np.hstack( (np.array([self.xcenter-0.5*self.width]*npoints, dtype=np.float32).reshape((npoints, 1)), xmwall) )
            ypwall_coords = np.hstack( (ypwall, np.array([self.ycenter+0.5*self.height]*npoints, dtype=np.float32).reshape((npoints, 1))) )
            ymwall_coords = np.hstack( (ymwall, np.array([self.ycenter-0.5*self.height]*npoints, dtype=np.float32).reshape((npoints, 1))) )

            all_coords = np.vstack((xpwall_coords, xmwall_coords, ypwall_coords, ymwall_coords)).astype(np.float32)
        
        return all_coords

    def is_inside(self, coords):
        in_xwindow = (coords[:, 0] > self.xcenter - 0.5*self.width) & (coords[:, 0] < self.xcenter + 0.5*self.width)
        in_ywindow = (coords[:, 1] > self.ycenter - 0.5*self.height) & (coords[:, 1] < self.ycenter + 0.5*self.height)
        is_in = np.logical_and(in_xwindow, in_ywindow)
        return is_in

    def __str__(self):
        return f'rect({self.xcenter}, {self.ycenter}, {self.width}, {self.height})'

class geom_circ(geom):
    def __init__(self, xcenter, ycenter, radius):
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.radius = radius
        pass

    def generate_points(self, npoints, interior=False):
        if interior:
            rsq = np.random.uniform(0, self.radius^2, size=(npoints, 1)).astype(np.float32)
            r = np.sqrt(rsq)
            phis = np.random.uniform(0, 2.0 * np.math.pi,size=(npoints, 1)).astype(np.float32)
            xcoords = r*np.cos(phis) + self.xcenter
            ycoords = r*np.sin(phis) + self.ycenter

        else:
            phis = np.random.uniform(0, 2.0 * np.math.pi,size=(npoints, 1)).astype(np.float32)
            xcoords = self.radius*np.cos(phis) + self.xcenter
            ycoords = self.radius*np.sin(phis) + self.ycenter

        all_coords = np.hstack( (xcoords, ycoords)).astype(np.float32)
        return all_coords
        
    def is_inside(self, coords):
        dist_center = np.hypot(coords[:, 0]-self.xcenter, coords[:,1]-self.ycenter)
        is_in = dist_center<self.radius
        return is_in
           
    def __str__(self):
        return f'circ({self.xcenter}, {self.ycenter}, {self.radius})'

class geom_poly(geom):
    def __init__(self, vertices):
        self.vertices = vertices
        
    pass