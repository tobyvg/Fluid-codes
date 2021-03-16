import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import integrate
import matplotlib as mpl
import copy

def init_mpl(global_dpi,labelsize = 15.5,legendsize = 11.40, fontsize = 13,mat_settings = False): # 13
    if mat_settings:
        fontsize = 10
        labelsize = 13
    mpl.rcParams['figure.dpi']= global_dpi
    mpl.rc('axes', labelsize=labelsize)
    font = {'size'   : fontsize}#'family' : 'normal', #'weight' : 'bold'
    mpl.rc('font', **font)
    mpl.rc('legend',fontsize = legendsize)

def vis_mat(matrix,color = mpl.cm.nipy_spectral, plot = True, vmin = None, vmax = None, ranges = None, rounding = 3, colorbar = True,fig_ax = None,x_points = 9, y_points =9,x_rotation = -45,labels = True,return_ax = False):
    if fig_ax == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    if ranges == None:
        img = ax.imshow(matrix, cmap = color, extent=[0,matrix.shape[1],matrix.shape[0],0], vmin = vmin, vmax = vmax)
    else:
        x_max = max(np.abs(ranges[0]),np.abs(ranges[1]))
        y_max = max(np.abs(ranges[2]),np.abs(ranges[3]))
        x_round = rounding - int(np.log10(x_max)//1)
        y_round = rounding - int(np.log10(y_max)//1)
        img = ax.imshow(matrix,cmap = color, vmin = vmin, vmax = vmax,extent=[-1,1,-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0])])
        if len(matrix[0]) < x_points:
            ax.set_xticks(np.linspace(-1,1,x_points))
        else:
            ax.set_xticks(np.linspace(-1,1-2/len(matrix[0]),x_points))
        if len(matrix) < y_points:
            ax.set_yticks(np.linspace(-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0]),y_points))
        else:
            ax.set_yticks(np.linspace(-len(matrix)/len(matrix[0]),len(matrix)/len(matrix[0])-
                                      len(matrix)/len(matrix[0])* 2/len(matrix),y_points))
        if x_round > 0 and x_round < 2*rounding:
            x = np.round(np.linspace(ranges[0],ranges[1],x_points),x_round)
        else:
            x = np.linspace(ranges[0],ranges[1],x_points)
            x = [np.format_float_scientific(i,precision = rounding-1) for i in x]
        if y_round > 0 and y_round < 2*rounding:
            y = np.round(np.linspace(ranges[3],ranges[2],y_points),y_round)
        else:
            y = np.linspace(ranges[3],ranges[2],y_points)
            y = [np.format_float_scientific(i,precision = rounding-1) for i in y]
        y = np.flip(y)
        ax.set_xticklabels(x,rotation = x_rotation)
        ax.set_yticklabels(y)
    if colorbar:
        fig.colorbar(img,ax = ax)
    if not labels:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    if plot:
        plt.show()
    if not plot and return_ax:
        return ax

class Poly:
    def __init__(self,coeffs): ## Initialize polynomial with coefficients as numpy array: [5,2,3] -> 5 + 2 x + 3 x^2
        self.coeffs = np.array(coeffs)
        self.trim()
    def set_coeffs(self,coeffs): ## Change polynomial
        self.coeffs = np.array(coeffs)
        self.trim()
    def evaluate(self,x): ## Evaluate polynomial, works for single numbers as well as numpy arrays
        for i in range(len(self.coeffs)):
            if i == 0:
                y = self.coeffs[i]*x**i
            else:
                y += self.coeffs[i]*x**i
        return y
    def diff(self,times = 1): ## returns differentiated polynomial as Poly object
        cache = copy.deepcopy(self.coeffs)
        for i in range(times):
            for j in range(len(self.coeffs)-1):
                cache[j] = (j+1)*cache[j+1]
            cache[-1] = 0
        return Poly(cache)
    def printer(self): ## prints polynomial
        string = ''
        for i in range(len(self.coeffs)):
            if i == 0:
                string += str(self.coeffs[i])
            else:
                string +='+' + str(self.coeffs[i]) + 'x^' + str(i)
        print(string)
    def trim(self): ## removes zeros at the end of the coefficient array
        check = False
        index = None
        for i in range(1,len(self.coeffs)+1):
            if self.coeffs[-i] != 0 and not check:
                index = i
                check = True
        if index != 1 and index != None:
            self.coeffs = self.coeffs[:-(index-1)]
    def integ(self,times = 1): ## returns integrated polynomial as Poly object
        cache1 = copy.deepcopy(self.coeffs)
        for i in range(times):
            cache = np.zeros(cache1.shape[0]+1)
            for j in range(len(cache1)):
                cache[j+1] = 1/(j+1)*cache1[j]
            cache1 = copy.deepcopy(cache)
        return Poly(cache)

def PolyMult(pol1,pol2,m1=1,m2=1): ## Multiplies 2 polynomial objects weighted by m1 and m2 respectively
    c1 = m1*pol1.coeffs
    c2 = m2*pol2.coeffs
    cache = np.zeros(len(c1)+len(c2))
    for i in range(len(c1)):
        for j in range(len(c2)):
            cache[j+i] += c1[i]*c2[j]
    return Poly(cache)

def PolyAdd(pol1,pol2,m1=1,m2=1): ## Adds 2 polynomial objects multiplied by m1 and m2 respectively
    c1 = m1*copy.deepcopy(pol1.coeffs)
    c2 = m2*copy.deepcopy(pol2.coeffs)
    if len(c1) > len(c2):
        for i in range(len(c2)):
            c1[i] += c2[i]
        return Poly(c1)
    else:
        for i in range(len(c1)):
            c2[i] += c1[i]
        return Poly(c2)

def xtoxi(x,l,u):
    return  (x-l)/(u-l)


def Bernstein(index):
    assert index in [1,2,3]
    if index == 1:
        return Poly([1,-2,1])
    if index == 2:
        return Poly([0,2,-2])
    if index == 3:
        return Poly([0,0,1])

######### Finite volume code
def form_grid(n1,n2,xstart = 0,xend = 1,ystart = 0,yend = 1):
    x = np.linspace(xstart,xend,n1+1)
    y = np.linspace(ystart,yend,n2+1)
    Z = np.zeros((y.shape[0],x.shape[0],2))
    cache = y.copy()
    cache = np.flipud(cache)
    Z[:,:,0] = x
    Z = Z.T
    Z[1,:,:] = cache
    return Z.T





def basis(t1,t2,i1,i2,h1,h2): #### types: 0 constant, 1 linear, index: 0 before node, 1 after, h: element boundaries [x_i,x_{i+1}]
    polys = []
    for t,i,h in zip([t1,t2],[i1,i2],[h1,h2]):
        if t == 0:
            polys.append(Poly([1]))
        if t == 1:
            diff = h[1] - h[0]
            if i == 0:
                polys.append(Poly([-h[0]/diff,1/diff]))
            if i == 1:
                polys.append(Poly([h[1]/diff,-1/diff]))
    return polys

def v0(x,y):

    if np.round(y,5) >= 1:
        return np.array([1,0])
    else:
        return np.array([0,0])


def f(x,y):
    return np.array([0,0])

def grid_checker(dicto,index):
    checks = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    counter = 1
    for i in checks:
        if i in dicto[index]:
            if not np.isnan(dicto[index][i][0][0]):
                plt.scatter(dicto[index][i][0][0],dicto[index][i][0][1],marker = '$' + str(dicto[index][i][1]) + '$')
                counter += 1
    plt.show()


def index_calc(i,j,dim1,dim2):
    return i*dim2 + j

def rev_index_calc(k,dim1,dim2):
    i = k//dim2
    j = k - dim2*(k//dim2)
    return np.array([i,j])

def start_end(start,end):
    s = start[0]
    e = end[0]
    return min(s[0],e[0]),max(s[0],e[0]),min(s[1],e[1]),max(s[1],e[1])



def matrix_vector_generator(X,v0,f,f_zero = False):
    grid_points = np.prod(X.shape[0:2])
    dim1 = X.shape[0]
    dim2 = X.shape[1]

    xvdim1 = (dim1-1)
    xvdim2 = (dim2-2)

    v_offset = xvdim1*xvdim2

    yvdim1 = (dim1-2)
    yvdim2 = (dim2-1)

    M = np.zeros((grid_points,grid_points))
    t = np.zeros((grid_points))
    t_test = t.copy()
    C = np.zeros((grid_points,xvdim1*xvdim2+yvdim1*yvdim2))

    D = np.zeros(((dim1-1)*(dim2-1),xvdim1*xvdim2+yvdim1*yvdim2))
    gh = np.zeros((dim1-1)*(dim2-1))

    fh = np.zeros((xvdim1*xvdim2+yvdim1*yvdim2))

    indices = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    neighbors_mem = {}


    for i in indices:
        neighbors_mem[i] = (np.array([np.nan,np.nan]),np.nan)
    for i in range(dim1):
        for j in range(dim2):
            #################### Storage of all neighboring indices #############
            point = X[i,j]
            neighbors = copy.deepcopy(neighbors_mem)
            row = index_calc(i,j,dim1,dim2)
            neighbors[(0,0)] = (point, row)
            if i != 0:
                neighbors[(0,1)] = (X[i-1,j], index_calc(i-1,j,dim1,dim2))
                if j != dim2-1:
                    neighbors[(1,1)] = (X[i-1,j+1], index_calc(i-1,j+1,dim1,dim2))
                    neighbors[(1,0)] = (X[i,j+1], index_calc(i,j+1,dim1,dim2))
                if j != 0:
                    neighbors[(-1,1)] = (X[i-1,j-1], index_calc(i-1,j-1,dim1,dim2))
                    neighbors[(-1,0)] = (X[i,j-1], index_calc(i,j-1,dim1,dim2))
            if i != dim1 - 1:
                neighbors[(0,-1)] = (X[i+1,j], index_calc(i+1,j,dim1,dim2))
                if j != dim2-1:
                    neighbors[(1,-1)] = (X[i+1,j+1], index_calc(i+1,j+1,dim1,dim2))
                    neighbors[(1,0)] = (X[i,j+1], index_calc(i,j+1,dim1,dim2))
                if j != 0:
                    neighbors[(-1,-1)] = (X[i+1,j-1], index_calc(i+1,j-1,dim1,dim2))
                    neighbors[(-1,0)] = (X[i,j-1], index_calc(i,j-1,dim1,dim2))

            #####################################################################
            #####################################################################
            ######################################################################
            ######### # Quadrant 1: Bottom left: ##########################################
            ####################################################################
            #####################################################################

            ur = lambda sx,ex,sy,ey: basis(1,1,0,0,[sx,ex],[sy,ey])
            ul = lambda sx,ex,sy,ey: basis(1,1,1,0,[sx,ex],[sy,ey])
            dr = lambda sx,ex,sy,ey: basis(1,1,0,1,[sx,ex],[sy,ey])
            dl = lambda sx,ex,sy,ey: basis(1,1,1,1,[sx,ex],[sy,ey])

            dx = lambda sx,ex,sy,ey: basis(1,0,1,1,[sx,ex],[sy,ey])
            ux = lambda sx,ex,sy,ey: basis(1,0,0,1,[sx,ex],[sy,ey])

            dy = lambda sx,ex,sy,ey: basis(0,1,1,1,[sx,ex],[sy,ey])
            uy = lambda sx,ex,sy,ey: basis(0,1,1,0,[sx,ex],[sy,ey])


            checks = [neighbors[(-1,-1)],neighbors[(-1,1)],neighbors[(1,1)],neighbors[(1,-1)]]

            ### BL -> TL -> TR -> BR

            useds = [[(-1,-1),(-1,0),(0,0),(0,-1)],
                      [(-1,0),(-1,1),(0,1),(0,0)],
                      [(0,0),(0,1),(1,1),(1,0)],
                      [(0,-1),(0,0),(1,0),(1,-1)]]
            usedsx = [[(-1,-1),(0,-1)],[(-1,0),(0,0)],[(0,0),(1,0)],[(0,-1),(1,-1)]]
            usedsy = [[(0,-1),(0,0)],[(0,0),(0,1)], [(1,0),(1,1)],[(1,-1),(1,0)]]
            B00fs = [ur,dr,dl,ul]
            quadrants = [0,1,2,3]

            for check,used,usedx,usedy,B00f,quadrant in zip(checks,useds,usedsx,usedsy,B00fs,quadrants):
                if not np.isnan(check[0][0]):
                    ### Integral 1 ########################################################
                    #########################################################################
                    sx,ex,sy,ey = start_end(check,neighbors[(0,0)])

                    #print(neighbors[(0,0)][0])
                    #print(check[0])


                    B00 = B00f(sx,ex,sy,ey)

                    b_used = [basisfunct(sx,ex,sy,ey) for basisfunct in [dl,ul,ur,dr]]
                    #used = [(-1,-1),(-1,0),(0,0),(0,-1)]


                    for u,b in zip(used,b_used):
                        I = ([PolyMult(B00[i],b[i]).integ() for i in range(2)])
                        O = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])

                        M[row,neighbors[u][1]] += O

                    ## t integrals 1 ######################################################

                    if i == 0:
                        if quadrant == 0 or quadrant == 3:
                            funct = lambda x: -B00[0].evaluate(x)*B00[1].evaluate(ey)*v0(x,ey)[0]
                            I = integrate.quad(funct,sx,ex)[0]
                            t_test[row] += I
                    if i == dim1 - 1:
                        if quadrant == 1 or quadrant == 2:
                            funct = lambda x: B00[0].evaluate(x)*B00[1].evaluate(sy)*v0(x,sy)[0]
                            I = integrate.quad(funct,sx,ex)[0]
                            t_test[row] += I

                    ## t integrals 1 ######################################################

                    if j == 0:
                        if quadrant == 2 or quadrant == 3:
                            funct = lambda y: -B00[0].evaluate(sx)*B00[1].evaluate(y)*v0(sx,y)[1]
                            I = integrate.quad(funct,sy,ey)[0]
                            t_test[row] += I
                    if j == dim2 - 1:
                        if quadrant == 0 or quadrant == 1:
                            funct = lambda y: B00[0].evaluate(ex)*B00[1].evaluate(y)*v0(ex,y)[1]
                            I = integrate.quad(funct,sy,ey)[0]
                            t_test[row] += I


                    #####################################################################################
                    #######################################################################
                    ### Integral 2,1 ################

                    dB00 = [B00[0],PolyMult(Poly([-1]),B00[1].diff())]
                    b_used = [basisfunct(sx,ex,sy,ey) for basisfunct in [dx,ux]]
                    #used = [(-1,-1),(0,-1)]
                    for u,b in zip(usedx,b_used):
                        M_index = neighbors[u][1]
                        ci,cj = rev_index_calc(M_index,dim1,dim2)
                        col = index_calc(ci-1,cj-1,xvdim1,xvdim2)
                        if ci != 0 and cj != 0 and cj != dim2 -1:
                            I = ([PolyMult(dB00[i],b[i]).integ() for i in range(2)])
                            O = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                            C[row,col] += O
                        elif ci != 0 and cj == 0 or cj == dim2-1:
                            funct = [PolyMult(dB00[0],b[0]),PolyMult(dB00[1],Poly([v0(neighbors[u][0][0],(sy+ey)/2)[0]]))]
                            I = [i.integ() for i in funct]
                            O = np.prod([i.evaluate(e)-i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                            t[row] -= O






                    ### Integral 2.2 ##################
                    dB00 = [B00[0].diff(),B00[1]]
                    b_used = [basisfunct(sx,ex,sy,ey) for basisfunct in [dy,uy]]
                    #used = [(0,-1),(0,0)]
                    for u,b in zip(usedy,b_used):
                        M_index = neighbors[u][1]
                        ci,cj = rev_index_calc(M_index,dim1,dim2)
                        col = index_calc(ci-1,cj-1,yvdim1,yvdim2)+v_offset
                        if ci != 0 and ci != dim1 -1 and cj != 0:
                            I = ([PolyMult(dB00[i],b[i]).integ() for i in range(2)])
                            O = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                            C[row,col] += O
                        elif cj != 0 and ci == 0 or ci == dim1-1:
                            funct = [PolyMult(dB00[0],Poly([v0((sx+ex)/2,neighbors[u][0][1])[1]])),PolyMult(dB00[1],b[1])]
                            I = [i.integ() for i in funct]
                            O = np.prod([i.evaluate(e)-i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                            t[row] -= O

                    ############ f integrals ########################
                    if not f_zero:
                        if i != 0 and j != 0 and j != dim2 -1 and (quadrant ==1 or quadrant == 2):
                            w_row = index_calc(i-1,j-1,xvdim1,xvdim2)
                            W = [B00[0],Poly([1])]
                            Wf = lambda y,x: W[0].evaluate(x)*W[1].evaluate(y)*(f(x,y)[0])
                            I = integrate.dblquad(Wf,sx,ex,lambda x: sy, lambda x: ey)[0]
                            fh[w_row] += I




                        if i != 0 and i != dim1 -1 and j != 0 and (quadrant == 0 or quadrant == 1):
                            w_row = index_calc(i-1,j-1,yvdim1,yvdim2) + v_offset
                            W = [Poly([1]),B00[1]]
                            Wf = lambda y,x: W[0].evaluate(x)*W[1].evaluate(y)*(f(x,y)[1])
                            I = integrate.dblquad(Wf,sx,ex,lambda x: sy, lambda x: ey)[0]
                            fh[w_row] += I




                    ############ q integrals #########################
                    if quadrant == 1:
                        ### Last integral.1
                        if i != 0 and j != 0:
                            B = basis(0,0,0,0,[sx,ex],[sy,ey])
                            #q_indx = np.array([i-1,j-1])
                            q_row = index_calc(i-1,j-1,dim1-1,dim2-1)
                            b_used = [basisfunct(sx,ex,sy,ey) for basisfunct in [dx,ux]]

                            ##############################3
                            for u,b in zip(usedx,b_used):
                                M_index = neighbors[u][1]
                                ci,cj = rev_index_calc(M_index,dim1,dim2)
                                col = index_calc(ci-1,cj-1,xvdim1,xvdim2)
                                if ci != 0 and cj != 0 and cj != dim2 -1:
                                    funct = [PolyMult(B[0],b[0].diff()),PolyMult(B[1],b[1])]
                                    I = [i.integ() for i in funct]
                                    O = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                                    D[q_row,col] += O
                                elif ci != 0 and cj == 0 or cj == dim2-1:
                                    funct = [PolyMult(B[0],b[0].diff()),PolyMult(B[1],Poly([v0(neighbors[u][0][0],(sy+ey)/2)[0]]))]
                                    I = [i.integ() for i in funct]
                                    O = np.prod([i.evaluate(e)-i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                                    gh[q_row] -= O

                            #########################################################3
                        ### Last_integral.2 ##################
                            b_used = [basisfunct(sx,ex,sy,ey) for basisfunct in [dy,uy]]
                            #used = [(0,-1),(0,0)]

                            for u,b in zip(usedy,b_used):
                                M_index = neighbors[u][1]
                                ci,cj = rev_index_calc(M_index,dim1,dim2)
                                col = index_calc(ci-1,cj-1,yvdim1,yvdim2)+v_offset
                                if ci != 0 and ci != dim1 -1 and cj != 0:
                                    funct = [PolyMult(B[0],b[0]),PolyMult(B[1],b[1].diff())]
                                    I = [i.integ() for i in funct]
                                    O = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                                    D[q_row,col] += O
                                elif cj != 0 and ci == 0 or ci == dim1-1:
                                    funct = [PolyMult(B[0],Poly([v0((sx+ex)/2,neighbors[u][0][1])[1]])),PolyMult(B[1],b[1].diff())]
                                    I = [i.integ() for i in funct]
                                    O = np.prod([i.evaluate(e)-i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])
                                    gh[q_row] -= O

    A = -C.T
    G = -D.T
    #G = -D.T
    return (M,C,A,G,D),(t+t_test,fh,gh)

def stream_function(X,v0,w):
    grid_points = np.prod(X.shape[0:2])
    dim1 = X.shape[0]
    dim2 = X.shape[1]

    w_flat = w.flatten()

    M = np.zeros((grid_points,grid_points))
    t = np.zeros((grid_points))

    t_test = t.copy()

    indices = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    neighbors_mem = {}


    for i in indices:
        neighbors_mem[i] = (np.array([np.nan,np.nan]),np.nan)
    for i in range(dim1):
        for j in range(dim2):
            #################### Storage of all neighboring indices #############
            point = X[i,j]
            neighbors = copy.deepcopy(neighbors_mem)
            row = index_calc(i,j,dim1,dim2)
            neighbors[(0,0)] = (point, row)
            if i != 0:
                neighbors[(0,1)] = (X[i-1,j], index_calc(i-1,j,dim1,dim2))
                if j != dim2-1:
                    neighbors[(1,1)] = (X[i-1,j+1], index_calc(i-1,j+1,dim1,dim2))
                    neighbors[(1,0)] = (X[i,j+1], index_calc(i,j+1,dim1,dim2))
                if j != 0:
                    neighbors[(-1,1)] = (X[i-1,j-1], index_calc(i-1,j-1,dim1,dim2))
                    neighbors[(-1,0)] = (X[i,j-1], index_calc(i,j-1,dim1,dim2))
            if i != dim1 - 1:
                neighbors[(0,-1)] = (X[i+1,j], index_calc(i+1,j,dim1,dim2))
                if j != dim2-1:
                    neighbors[(1,-1)] = (X[i+1,j+1], index_calc(i+1,j+1,dim1,dim2))
                    neighbors[(1,0)] = (X[i,j+1], index_calc(i,j+1,dim1,dim2))
                if j != 0:
                    neighbors[(-1,-1)] = (X[i+1,j-1], index_calc(i+1,j-1,dim1,dim2))
                    neighbors[(-1,0)] = (X[i,j-1], index_calc(i,j-1,dim1,dim2))

            #####################################################################
            #####################################################################
            ######################################################################
            ######### # Quadrant 1: Bottom left: ##########################################
            ####################################################################
            #####################################################################

            ur = lambda sx,ex,sy,ey: basis(1,1,0,0,[sx,ex],[sy,ey])
            ul = lambda sx,ex,sy,ey: basis(1,1,1,0,[sx,ex],[sy,ey])
            dr = lambda sx,ex,sy,ey: basis(1,1,0,1,[sx,ex],[sy,ey])
            dl = lambda sx,ex,sy,ey: basis(1,1,1,1,[sx,ex],[sy,ey])

            dx = lambda sx,ex,sy,ey: basis(1,0,1,1,[sx,ex],[sy,ey])
            ux = lambda sx,ex,sy,ey: basis(1,0,0,1,[sx,ex],[sy,ey])

            dy = lambda sx,ex,sy,ey: basis(0,1,1,1,[sx,ex],[sy,ey])
            uy = lambda sx,ex,sy,ey: basis(0,1,1,0,[sx,ex],[sy,ey])


            checks = [neighbors[(-1,-1)],neighbors[(-1,1)],neighbors[(1,1)],neighbors[(1,-1)]]

            ### BL -> TL -> TR -> BR

            useds = [[(-1,-1),(-1,0),(0,0),(0,-1)],
                      [(-1,0),(-1,1),(0,1),(0,0)],
                      [(0,0),(0,1),(1,1),(1,0)],
                      [(0,-1),(0,0),(1,0),(1,-1)]]
            usedsx = [[(-1,-1),(0,-1)],[(-1,0),(0,0)],[(0,0),(1,0)],[(0,-1),(1,-1)]]
            usedsy = [[(0,-1),(0,0)],[(0,0),(0,1)], [(1,0),(1,1)],[(1,-1),(1,0)]]
            B00fs = [ur,dr,dl,ul]
            quadrants = [0,1,2,3]

            for check,used,usedx,usedy,B00f,quadrant in zip(checks,useds,usedsx,usedsy,B00fs,quadrants):
                if not np.isnan(check[0][0]):
                    ### Integral 1 ########################################################
                    #########################################################################
                    sx,ex,sy,ey = start_end(check,neighbors[(0,0)])

                    #print(neighbors[(0,0)][0])
                    #print(check[0])


                    B00 = B00f(sx,ex,sy,ey)

                    b_used = [basisfunct(sx,ex,sy,ey) for basisfunct in [dl,ul,ur,dr]]
                    #used = [(-1,-1),(-1,0),(0,0),(0,-1)]


                    for u,b in zip(used,b_used):
                        I1 = [PolyMult(B00[0].diff(),b[0].diff()).integ(),PolyMult(B00[1],b[1]).integ()]
                        I2 = [PolyMult(B00[0],b[0]).integ(),PolyMult(B00[1].diff(),b[1].diff()).integ()]
                        O1 = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I1,[sx,sy],[ex,ey])])
                        O2 = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I2,[sx,sy],[ex,ey])])
                        M[row,neighbors[u][1]] +=  O2 + O1

                        I = ([PolyMult(B00[i],b[i]).integ() for i in range(2)])
                        O = np.prod([i.evaluate(e) - i.evaluate(s) for i,s,e in zip(I,[sx,sy],[ex,ey])])

                        t[row] += w_flat[neighbors[u][1]]*O


                    ## t integrals 1 ######################################################

                    if i == 0:
                        if quadrant == 0 or quadrant == 3:
                            funct = lambda x: B00[0].evaluate(x)*B00[1].evaluate(ey)*v0(x,ey)[0]
                            I = integrate.quad(funct,sx,ex)[0]
                            t_test[row] += I
                    if i == dim1 - 1:
                        if quadrant == 1 or quadrant == 2:
                            funct = lambda x: -B00[0].evaluate(x)*B00[1].evaluate(sy)*v0(x,sy)[0]
                            I = integrate.quad(funct,sx,ex)[0]
                            t_test[row] += I

                    ## t integrals 1 ######################################################

                    if j == 0:
                        if quadrant == 2 or quadrant == 3:
                            funct = lambda y: B00[0].evaluate(sx)*B00[1].evaluate(y)*v0(sx,y)[1]
                            I = integrate.quad(funct,sy,ey)[0]
                            t_test[row] += I
                    if j == dim2 - 1:
                        if quadrant == 0 or quadrant == 1:
                            funct = lambda y: -B00[0].evaluate(ex)*B00[1].evaluate(y)*v0(ex,y)[1]
                            I = integrate.quad(funct,sy,ey)[0]
                            t_test[row] += I



    return M,t + t_test


def matrix_vector_assembler(mats,vecs,normalize_p = True, visualize_matrices = False):
    M,C,A,G,D = mats
    vec = np.concatenate(vecs)
    if normalize_p:
        matrix = np.zeros((M.shape[0]+A.shape[0]+D.shape[0]+1,M.shape[1]+C.shape[1] + G.shape[1]))
    else:
        matrix = np.zeros((M.shape[0]+A.shape[0]+D.shape[0],M.shape[1]+C.shape[1] + G.shape[1]))

    matrix[:M.shape[0],:M.shape[1]] = M
    matrix[M.shape[0]:M.shape[0] + A.shape[0],:A.shape[1]] = A

    matrix[:C.shape[0],M.shape[1]:M.shape[1] + C.shape[1]] =C

    matrix[M.shape[0] + A.shape[0]:M.shape[0] + A.shape[0]+D.shape[0],M.shape[1]:M.shape[1] +D.shape[1]] = D

    matrix[M.shape[0]:M.shape[0]+G.shape[0],M.shape[1] + C.shape[1]:M.shape[1]+C.shape[1] + G.shape[1]] = G

    matrix[-1,M.shape[1] + C.shape[1]:M.shape[1]+C.shape[1] + G.shape[1]] = np.ones(G.shape[1])

    if visualize_matrices:
        vis_mat(M,plot = False)
        plt.title('$\mathbf{M}$')
        plt.show()
        vis_mat(C,plot = False)
        plt.title('$\mathbf{C}$')
        plt.show()
        vis_mat(A,plot = False)
        plt.title('$\mathbf{A}$')
        plt.show()
        vis_mat(G,plot = False)
        plt.title('$\mathbf{G}$')
        plt.show()
        vis_mat(D,plot = False)
        plt.title('$\mathbf{D}$')
        plt.show()
        vis_mat(matrix,plot = False)
        plt.title('Full system')
        plt.show()

    del M
    del C
    del A
    del G
    del D

    if normalize_p:
        vec = np.concatenate((vec,np.array([0])))
        return matrix.T@ matrix,matrix.T@vec

    else:
        return matrix,vec



def organize_sol(grid,sol,v0):
    dim1 = grid.shape[0]
    dim2 = grid.shape[1]
    w = sol[:dim1*dim2].reshape(dim1,dim2)
    offset = dim1*dim2
    x_offset = (dim1-1)*(dim2-2)
    vx = sol[offset:offset + x_offset]
    vx = vx.reshape((dim1-1,dim2-2))
    offset = offset + x_offset
    y_offset = (dim1-2)*(dim2-1)
    vy = sol[offset:offset+y_offset]
    vy = vy.reshape((dim1-2,dim2-1))
    p = sol[offset+y_offset:]
    p = p.reshape(dim1-1,dim2-1)



    return w,vx,vy,p




def process_wv(grid,w,vx,vy,v0):

    #########################################
    left_vx = np.array([v0(0,x)[0] for x in grid[1:,0,1]])
    right_vx= np.array([v0(1,x)[0] for x in grid[1:,-1,1]])
    vx_full = np.zeros((vx.shape[0],vx.shape[1]+2))
    vx_full[:,0] = left_vx
    vx_full[:,-1] = right_vx
    vx_full[:,1:-1] = vx
    #########################################

    #########################################
    bottom_vy = np.array([v0(x,0)[1]for x in grid[-1,1:,0]])
    top_vy = np.array([v0(x,1)[1] for x in grid[0,1:,0] ])
    vy_full = np.zeros((vy.shape[0]+2,vy.shape[1]))
    vy_full[0,:] = top_vy
    vy_full[-1,:] = bottom_vy
    vy_full[1:-1,:] = vy
    #########################################

    dx = lambda sx,ex,sy,ey: basis(1,0,1,1,[sx,ex],[sy,ey])
    ux = lambda sx,ex,sy,ey: basis(1,0,0,1,[sx,ex],[sy,ey])

    dvx = np.zeros((vx_full.shape[0],vx_full.shape[1]-1))
    vx = dvx.copy()

    for i in range(grid.shape[0]-1):
        for j in range(grid.shape[1]-1):
            ul = grid[i,j]
            ur = grid[i,j+1]

            c1, c2 = vx_full[i,j], vx_full[i,j+1]

            sx,ex = ul[0],ur[0]
            center = (ex+sx)/2

            f1,f2 = dx(sx,ex,9999999,999999), ux(sx,ex,999999,999999)

            dvx[i,j] = c1*f1[0].diff().evaluate(center) + c2*f2[0].diff().evaluate(center)
            vx[i,j] = c1*f1[0].evaluate(center) + c2*f2[0].evaluate(center)


    dy = lambda sx,ex,sy,ey: basis(0,1,1,1,[sx,ex],[sy,ey])
    uy = lambda sx,ex,sy,ey: basis(0,1,1,0,[sx,ex],[sy,ey])

    dvy = np.zeros((vy_full.shape[0]-1,vy_full.shape[1]))
    vy = dvy.copy()

    for i in range(grid.shape[1]-1):
        for j in range(grid.shape[0]-1):
            ur = grid[j,i]
            dr = grid[j+1,i]

            c1, c2 = vy_full[j+1,i], vy_full[j,i]

            sy,ey = dr[1],ur[1]


            center = (sy+ey)/2



            f1,f2 = dy(99999999,99999999,sy,ey), uy(9999999,999999,sy,ey)

            dvy[j,i] = c1*f1[1].diff().evaluate(center) + c2*f2[1].diff().evaluate(center)
            vy[j,i] = c1*f1[1].evaluate(center) + c2*f2[1].evaluate(center)
    #vis_mat(dvy,color = 'brg')

    fur = lambda sx,ex,sy,ey: basis(1,1,0,0,[sx,ex],[sy,ey])
    ful = lambda sx,ex,sy,ey: basis(1,1,1,0,[sx,ex],[sy,ey])
    fdr = lambda sx,ex,sy,ey: basis(1,1,0,1,[sx,ex],[sy,ey])
    fdl = lambda sx,ex,sy,ey: basis(1,1,1,1,[sx,ex],[sy,ey])


    w_center = np.zeros((grid.shape[0]-1,grid.shape[1]-1))
    for i in range(grid.shape[0]-1):
        for j in range(grid.shape[1]-1):
            ul = grid[i,j]
            ur = grid[i,j+1]
            dr = grid[i+1,j+1]
            dl = grid[i+1,j]

            c1,c2,c3,c4 = w[i+1,j],w[i,j],w[i,j+1],w[i+1,j+1]

            sx,sy,ex,ey = dl[0],dl[1],ur[0],ur[1]

            centerx = (ex+sx)/2
            centery = (sy+ey)/2


            f1,f2,f3,f4 = [f(sx,ex,sy,ey) for f in [fdl,ful,fur,fdr]]

            w_center[i,j] = np.sum([c*f[0].evaluate(centerx)*f[1].evaluate(centery) for c,f in zip([c1,c2,c3,c4],[f1,f2,f3,f4])])
    return w_center, vx,vy,dvx + dvy


def process_psi(grid,psi):
    fur = lambda sx,ex,sy,ey: basis(1,1,0,0,[sx,ex],[sy,ey])
    ful = lambda sx,ex,sy,ey: basis(1,1,1,0,[sx,ex],[sy,ey])
    fdr = lambda sx,ex,sy,ey: basis(1,1,0,1,[sx,ex],[sy,ey])
    fdl = lambda sx,ex,sy,ey: basis(1,1,1,1,[sx,ex],[sy,ey])


    psi_center = np.zeros((grid.shape[0]-1,grid.shape[1]-1))
    for i in range(grid.shape[0]-1):
        for j in range(grid.shape[1]-1):
            ul = grid[i,j]
            ur = grid[i,j+1]
            dr = grid[i+1,j+1]
            dl = grid[i+1,j]

            c1,c2,c3,c4 = psi[i+1,j],psi[i,j],psi[i,j+1],psi[i+1,j+1]

            sx,sy,ex,ey = dl[0],dl[1],ur[0],ur[1]

            centerx = (ex+sx)/2
            centery = (sy+ey)/2


            f1,f2,f3,f4 = [f(sx,ex,sy,ey) for f in [fdl,ful,fur,fdr]]

            psi_center[i,j] = np.sum([c*f[0].evaluate(centerx)*f[1].evaluate(centery) for c,f in zip([c1,c2,c3,c4],[f1,f2,f3,f4])])
    return psi_center


def Stokes_FEM(grid,v0,f ,visualize_matrices = False,xmin = 0,xmax = 1,ymin = 0,ymax = 1,f_zero = False):
    matrices,vecs = matrix_vector_generator(grid,v0,f,f_zero = f_zero)
    matrix,vec = matrix_vector_assembler(matrices,vecs, visualize_matrices = visualize_matrices)
    sol = np.linalg.solve(matrix,vec)
    del matrix
    del vec
    w,vx,vy,p = organize_sol(grid,sol,v0)
    matrix,vec = stream_function(grid,v0,w)
    ###################### Stream function ##################
    LS_matrix = np.concatenate([matrix, np.array([np.ones(matrix.shape[1])])])
    LS_vec = np.concatenate([vec,np.array([0])])
    psi = np.linalg.solve(LS_matrix.T @ LS_matrix,LS_matrix.T @ LS_vec).reshape(grid.shape[0],grid.shape[1])
    psi = process_psi(grid,psi)
    #########################################################
    w,vx,vy,dv = process_wv(grid,w,vx,vy,v0)
    return w,vx,vy,p,dv,psi



    #return checker
