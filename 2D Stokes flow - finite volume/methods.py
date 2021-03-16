import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Visualization code
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


######### Finite volume code
def form_grid(x,y):
    Z = np.zeros((y.shape[0],x.shape[0],2))
    cache = y.copy()
    cache = np.flipud(cache)
    Z[:,:,0] = x
    Z = Z.T
    Z[1,:,:] = cache
    return Z.T

def create_mesh(ranges,J1,J2):
    x_range = (ranges[1] - ranges[0])/(J1)
    y_range = (ranges[3] - ranges[0])/(J2)
    x = np.linspace(ranges[0]+1/2*x_range,ranges[1]-1/2*x_range,J1)
    y = np.linspace(ranges[2]+1/2*y_range,ranges[3]-1/2*y_range,J2)
    return form_grid(x,y)

def vel_uniform_mesh(Z):
    dim1 = Z.shape[0]
    dim2 = Z.shape[1]
    trans_dict = {}
    coord_mat = [] ###########Save all coords
    for i in range(dim1):
        cache = []
        for j in range(dim2-1):
            cache.append((Z[i,j]+Z[i,j+1])/2)
        coord_mat.append(np.array(cache))
        if i != dim1 - 1:
            cache = []
            for j in range(dim2):
                cache.append((Z[i,j]+Z[i+1,j])/2)
            coord_mat.append(np.array(cache))
    coord_mat = np.array(coord_mat)
    #for i in range(dim1):
    #    for j in range(dim2):
    for i in range(dim1):
        for j in range(dim2):
            cache = []
            ###### boundary 1 ###########
            if i == dim1 - 1:
                cache.append(np.array([np.nan,np.nan]))
            else:
                cache.append(np.array([2*i+1,j]))

            ########## boundary 2 ################
            if j == dim2-1:
                cache.append(np.array([np.nan,np.nan]))
            else:
                cache.append(np.array([2*i,j]))

            ########## boundary 3 ################
            if i == 0:
                cache.append(np.array([np.nan,np.nan]))
            else:
                cache.append(np.array([2*i-1,j]))
            ########## boundary 4 ################
            if j == 0:
                cache.append(np.array([np.nan,np.nan]))
            else:
                cache.append(np.array([2*i,j-1]))
            trans_dict[(i,j)] = cache
    return coord_mat, trans_dict

def coord_calc(index1,index2,C):
    len1 = len(C[0])
    len2 = len(C[1])
    if index1//2 - index1/2 == 0:
        return int(index1/2*(len1+len2) + index2)
    else:
        return int(len1 + (index1-1)/2*(len1+len2) + index2)


def D_mat(Z,C, t_dict,v0,plot = False):
    h1 = (Z[0,1]-Z[0,0])[0]
    h2 = (Z[0,0]-Z[1,0])[1]
    dim1 = Z.shape[0]
    dim2 = Z.shape[1]
    prod = dim1*dim2
    total = len(np.concatenate(C))
    D = np.zeros((prod,total))
    g = np.zeros((prod))
    for i in range(dim1):
        for j in range(dim2):
            row = i*dim2 + j
            t = t_dict[(i,j)]


            ######## Boundary 1
            index = 0
            if np.isnan(t[index][0]):
                corr_c = C[t[2][0]][t[2][1]] - np.array([0,h2])
                g[row] += -(np.array([0,-1])@v0(corr_c))*h1
                if plot:
                    plt.scatter(corr_c[0],corr_c[1], c = 'blue')
            else:
                col = coord_calc(t[index][0],t[index][1],C)
                D[row,col] = -h1

            ######## Boundary 2
            index = 1
            if np.isnan(t[index][0]):

                corr_c =C[t[3][0]][t[3][1]] + np.array([h1,0])
                g[row] += -(np.array([1,0])@v0(corr_c))*h2
                if plot:
                    plt.scatter(corr_c[0],corr_c[1], c = 'blue')
            else:
                col = coord_calc(t[index][0],t[index][1],C)
                D[row,col] = h2


            ######### Boundary 3
            index = 2
            if np.isnan(t[index][0]):
                corr_c = C[t[0][0]][t[0][1]] + np.array([0,h2])
                g[row] += -(np.array([0,1])@v0(corr_c))*h1
                if plot:
                    plt.scatter(corr_c[0],corr_c[1], c = 'blue')
            else:
                col = coord_calc(t[index][0],t[index][1],C)
                D[row,col] =  h1


            ######### Boundary 4
            index = 3
            if np.isnan(t[index][0]):
                corr_c = C[t[1][0]][t[1][1]] - np.array([h1,0])
                g[row] += -(np.array([-1,0])@v0(corr_c))*h2
                if plot:
                    plt.scatter(corr_c[0],corr_c[1], c = 'blue')
            else:
                col = coord_calc(t[index][0],t[index][1],C)
                D[row,col] =  -h2
    if plot:
        plt.show()
    return D, g

def coord_calc_p(index1,index2,Z):
    len2 = Z.shape[1]
    return index1*len2 + index2

def A_G_mat(Z,C, t_dict,v0,f,plot = False):
    h1 = (Z[0,1]-Z[0,0])[0]
    h2 = (Z[0,0]-Z[1,0])[1]
    dim1 = Z.shape[0]
    dim2 = Z.shape[1]
    prod = dim1*dim2
    total = len(np.concatenate(C))
    control_x = dim1*(dim2-1)
    control_y = dim2*(dim1-1)
    A = np.zeros((control_x + control_y,total))
    G = np.zeros((control_x + control_y,prod))
    f_vec = np.zeros((control_x + control_y))

    ############ index1 #######################3
    for i in range(dim1):
        for j in range(dim2-1):
            row = i*(dim2-1) + j
            volumes = [(i,j),(i,j+1)]
            vels = [t_dict[x] for x in volumes]




            ######### G matrix #############
            G[row, coord_calc_p(volumes[0][0],volumes[0][1],Z)] += h2
            G[row, coord_calc_p(volumes[1][0],volumes[1][1],Z)] += -h2
            if plot:
                plt.scatter(Z[volumes[0]][0],Z[volumes[0]][1], c = 'blue',s = 10,zorder = 1)
                plt.scatter(Z[volumes[1]][0],Z[volumes[1]][1],c = 'red', s = 5,zorder = 2)
            f_vec[row] += h1*h2*f((Z[volumes[0][0],volumes[0][1]] + Z[volumes[1][0],volumes[1][1]])/2)[0]

            left_coord = Z[volumes[0][0],volumes[0][1]]


            ######### boundary 1 ############# - alt

            if not np.isnan(vels[0][0][0]):

                A[row,coord_calc(vels[0][1][0],vels[0][1][1],C)] += -h1/h2
                index = (i+1,j)
                vel = t_dict[index]

                A[row,coord_calc(vel[1][0],vel[1][1],C)] += h1/h2
            else:
                A[row,coord_calc(vels[0][1][0],vels[0][1][1],C)] += -2*h1/h2
                c = left_coord[1] - 1/2*h2
                corr_c = np.array([left_coord[0] + 1/2*h1,c])
                f_vec[row] += - 2*v0(corr_c)[0]*h1/h2
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')
            ######### boundary 2 ############# +
            A[row,coord_calc(vels[0][1][0],vels[0][1][1],C)] += -h2/h1
            if not np.isnan(vels[1][1][0]):

                A[row,coord_calc(vels[1][1][0],vels[1][1][1],C)] += h2/h1
            else:
                c = left_coord[0] + 3/2*h1
                corr_c = np.array([c,left_coord[1]])
                f_vec[row] += - v0(corr_c)[0]*h2/h1
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')

            ######### boundary 3 ############# + alt

            if not np.isnan(vels[0][2][0]):

                A[row,coord_calc(vels[0][1][0],vels[0][1][1],C)] += -h1/h2
                index = (i-1,j)
                vel = t_dict[index]
                A[row,coord_calc(vel[1][0],vel[1][1],C)] += h1/h2
            else:
                A[row,coord_calc(vels[0][1][0],vels[0][1][1],C)] += -2*h1/h2
                c = left_coord[1] + 1/2*h2
                corr_c = np.array([left_coord[0] + 1/2*h1,c])
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')
                f_vec[row] += - 2*v0(corr_c)[0]*h1/h2

            ######### boundary 4 ############# -
            A[row,coord_calc(vels[0][1][0],vels[0][1][1],C)] += -h2/h1
            if not np.isnan(vels[0][3][0]):

                A[row,coord_calc(vels[0][3][0],vels[0][3][1],C)] += h2/h1
            else:
                c = left_coord[0] - 1/2*h1
                corr_c = np.array([c,left_coord[1]])
                f_vec[row] += - v0(corr_c)[0]*h2/h1
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')
    if plot:
        plt.show()

    ############ index2 #######################3
    for i in range(dim1-1):
        for j in range(dim2):
            row = control_x + i*(dim2) + j
            volumes = [(i,j),(i+1,j)]
            vels = [t_dict[x] for x in volumes]

            ######### G matrix #############
            G[row, coord_calc_p(volumes[0][0],volumes[0][1],Z)] += -h1
            G[row, coord_calc_p(volumes[1][0],volumes[1][1],Z)] += h1
            f_vec[row] += h1*h2*f((Z[volumes[0][0],volumes[0][1]] + Z[volumes[1][0],volumes[1][1]])/2)[1]
            if plot:
                plt.scatter(Z[volumes[0]][0],Z[volumes[0]][1], c = 'blue',s = 10,zorder = 1)
                plt.scatter(Z[volumes[1]][0],Z[volumes[1]][1],c = 'red', s = 5,zorder = 2)

            left_coord = Z[volumes[0][0],volumes[0][1]]

            ######### boundary 4 ############# - alt
            if not np.isnan(vels[0][3][0]):
                A[row,coord_calc(vels[0][0][0],vels[0][0][1],C)] += -h2/h1
                index = (i,j-1)
                vel = t_dict[index]
                A[row,coord_calc(vel[0][0],vel[0][1],C)] += h2/h1
            else:
                A[row,coord_calc(vels[0][0][0],vels[0][0][1],C)] += -2*h2/h1
                c = left_coord[0] - 1/2*h1
                corr_c = np.array([c,left_coord[1] - 1/2*h2])
                f_vec[row] += - 2*v0(corr_c)[1]*h2/h1
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')


            ######### boundary 3 ############# +
            A[row,coord_calc(vels[0][0][0],vels[0][0][1],C)] += -h1/h2
            if not np.isnan(vels[0][2][0]):
                A[row,coord_calc(vels[0][2][0],vels[0][2][1],C)] += h1/h2
            else:
                c = left_coord[1] + 1/2*h2
                corr_c = np.array([left_coord[0],c])
                f_vec[row] += - v0(corr_c)[1]*h1/h2
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')

            ######### boundary 2 ############# + alt
            if not np.isnan(vels[0][1][0]):
                A[row,coord_calc(vels[0][0][0],vels[0][0][1],C)] += -h2/h1
                index = (i,j+1)
                vel = t_dict[index]
                A[row,coord_calc(vel[0][0],vel[0][1],C)] += h2/h1
            else:
                A[row,coord_calc(vels[0][0][0],vels[0][0][1],C)] += -2*h2/h1
                c = left_coord[0] + 1/2*h1
                corr_c = np.array([c,left_coord[1] - 1/2*h2])
                f_vec[row] += - 2*v0(corr_c)[1]*h2/h1
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')



            ######### boundary 1 ############# -
            A[row,coord_calc(vels[0][0][0],vels[0][0][1],C)] += -h1/h2
            if not np.isnan(vels[1][0][0]):
                A[row,coord_calc(vels[1][0][0],vels[1][0][1],C)] += h1/h2
            else:
                c = left_coord[1] - 3/2*h2
                corr_c = np.array([left_coord[0],c])
                f_vec[row] += - v0(corr_c)[1]*h1/h2
                if plot:
                    plt.scatter(corr_c[0],corr_c[1],c = 'green')
    if plot:
        plt.show()

    return A,G,f_vec


def assemble(A,G,D,f_vec,g_vec, comp_cond = True):

    vec = np.concatenate([f_vec,g_vec])

    cache = np.zeros((D.shape[0],G.shape[1]))
    m1 = np.concatenate([A,G],axis = 1)
    m2 = np.concatenate([D,cache],axis = 1)
    M = np.concatenate([m1,m2])
    if comp_cond:
        zeros = np.zeros(len(f_vec))
        ones = np.ones(len(g_vec))
        concat = np.concatenate([zeros,ones])
        vec = np.concatenate([vec,np.array([0])])
        M = np.concatenate([M,np.array([concat])])
    return M, vec

def organize(sol,Z,C, t_dict,v0):
    h1 = (Z[0,1]-Z[0,0])[0]
    h2 = (Z[0,0]-Z[1,0])[1]
    total = len(np.concatenate(C))
    press = sol[total:].reshape((Z.shape[0],Z.shape[1]))
    cache = []
    counter = 0
    for i in C:
        length = len(i)
        cache.append(sol[counter:counter + length])
        counter += length
    vel = np.array(cache)
    corr_vel = np.zeros(Z.shape)
    for i in range(corr_vel.shape[0]):
        for j in range(corr_vel.shape[1]):
            indexes = t_dict[(i,j)]
            cache1 = 0
            cache0 = 0

            ind = 0
            cell_coord = Z[i,j]

            if not np.isnan(indexes[ind][0]):
                cache1 += vel[indexes[ind][0]][indexes[ind][1]]
            else:
                cache1 += v0(cell_coord - np.array([0,1/2*h2]))[1]

            ind = 1
            if not np.isnan(indexes[ind][0]):
                cache0 += vel[indexes[ind][0]][indexes[ind][1]]
            else:
                cache0 += v0(cell_coord + np.array([1/2*h1,0]))[0]
            ind = 2
            if not np.isnan(indexes[ind][0]):
                cache1 += vel[indexes[ind][0]][indexes[ind][1]]
            else:
                cache1 += v0(cell_coord + np.array([0,1/2*h2]))[1]
            ind = 3
            if not np.isnan(indexes[ind][0]):
                cache0 += vel[indexes[ind][0]][indexes[ind][1]]
            else:
                cache0 += v0(cell_coord - np.array([1/2*h1,0]))[0]
            corr_vel[i,j] = np.array([cache0/2,cache1/2])
    return press,corr_vel[:,:,0],corr_vel[:,:,1]

def Stokes_simulation(J1,J2,v0,f,domain = np.array([0,1,0,1]), visualize_matrices = False):
    z = create_mesh(domain,J1,J2)
    coord_mat,trans_dict = vel_uniform_mesh(z)
    D,g_vec = D_mat(z,coord_mat,trans_dict,v0)
    A,G,f_vec = A_G_mat(z,coord_mat, trans_dict,v0,f)
    M, vec = assemble(A,G,D,f_vec,g_vec,comp_cond = True)
    if visualize_matrices:
        vis_mat(D, plot = False,color = 'brg')
        plt.title('$\mathbf{D}_h$')
        plt.show()
        vis_mat(A, plot = False,color = 'brg')
        plt.title('$\mathbf{A}_h$')
        plt.show()
        vis_mat(G, plot = False,color = 'brg')
        plt.title('$\mathbf{G}_h$')
        plt.show()
        vis_mat(M, plot = False,color = 'brg')
        plt.title('$\mathbf{M}_h$')
        plt.show()
    sol = np.linalg.solve(M.T @ M,M.T @vec) #Least-squares
    press, vel_x,vel_y = organize(sol,z,coord_mat,trans_dict,v0)
    return z, press, vel_x,vel_y
