import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D

import glob
from matplotlib import pyplot as plt
import math
import time

#visual studio code ctrl + shift + B

def qm (q1, q2, q3, q4, p1, p2, p3, p4):
    r1 = q1*p1 - q2*p2 - q3*p3 - q4*p4
    r2 = q2*p1 + q1*p2 - q4*p3 + q3*p4
    r3 = q3*p1 + q4*p2 + q1*p3 - q2*p4
    r4 = q1*p1 - q3*p2 + q2*p3 + q1*p4
    return [r1, r2, r3, r4]

def rotq(r1, r2, r3, r4, q1, q2, q3, q4):
    temp = qm(r1, r2, r3, r4, q1, -q2, -q3, -q4)
    temp2 = qm(q1, q2, q3, q4, temp[0], temp[1], temp[2], temp[3])
    return temp2

def apply_T (tx, ty, tz, q1, q2, q3, q4, p1, p2, p3):
    # apply rotation first
    temp = rotq (0, p1, p2, p3, q1, q2, q3, q4) 
    rx = temp[1] + tx
    ry = temp[2] + ty
    rz = temp[3] + tz
    return [0, rx, ry, rz]


oxlist = []
oylist = []
ozlist = []

# original outputlist
output_list = [] 



with open(<Datapath>) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=" ")
    for row in csv_reader:
        
        t = float(row[0])
        #print( t )
        x = float(row[1])  
        y = float(row[2])  
        z = float(row[3])  
        q1 = float(row[4])  
        q2 = float(row[5])  
        q3 = float(row[6])  
        q4 = float(row[7])  
        oxlist.append(x)
        oylist.append(y)
        ozlist.append(z)
        output_list.append([t, x, y, z, q1, q2, q3, q4])

#CUT

for oitem in output_list:
   tempstring = str(oitem[0])
   oitem[0] = tempstring[0:7] #C8:0:6 / #C5: 0:7 /Euroc #0:12
   #print(oitem[0])



fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111, projection='3d') # Axe3D object

ax.plot(oxlist, oylist, ozlist, "b-")
plt.show()

gxlist = []
gylist = []
gzlist = []

gt_data_list = []


with open(<datapath>) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            ts = float(row[0])
            #print(ts)
            gx = float(row[1])
            gy = float(row[2])
            gz = float(row[3])
            gq1 = float(row[4])
            gq2 = float(row[5])
            gq3 = float(row[6])
            gq4 = float(row[7])
            gt_data = [ts, gx, gy, gz, gq1, gq2, gq3, gq4]
            gxlist.append(gx)
            gylist.append(gy)
            gzlist.append(gz)
            gt_data_list.append(gt_data)
            line_count += 1
    print(f'Processed {line_count} lines.')

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111, projection='3d') # Axe3D object

ax.plot(gxlist, gylist, gzlist, "y-")
plt.show()

#print(gt_data_list[0][0])

#for oitem in output_list:
   #print(oitem[0])
#for oitem in output_list:
   # print(oitem[0])

for gitem in gt_data_list:
  
   tempstring = str(gitem[0])
   gitem[0] = tempstring[0:7]
   #print(gitem)

print(output_list[0][0])
#
print("hi")
print(gt_data_list[0][0])

print("synch result")
synch_bin_list = []

for oinput in output_list:
    for ginput in  gt_data_list:
        if(oinput[0] == ginput[0]):
            g_match_data=[ginput[0],ginput[1],ginput[2],ginput[3],ginput[4],ginput[5],ginput[6],ginput[7]]
            synch_bin_list.append(g_match_data)
            #print("match")
            break


#print(synch_bin_list)


xxglist = []
yyglist = []
zzglist = []

for i in synch_bin_list:
    xxglist.append(i[1])
    yyglist.append(i[2])
    zzglist.append(i[3])

print()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot(xxglist, yyglist, zzglist, "black")
plt.show()




###pos####


dis_x=xxglist[0]-oxlist[0]
dis_y=yyglist[0]-oylist[0]
dis_z=zzglist[0]-ozlist[0]

for i in range(len(oxlist)):
    oxlist[i]=oxlist[i]+dis_x
    
for i in range(len(oylist)):
    oylist[i]=oylist[i]+dis_y
    
for i in range(len(ozlist)):
    ozlist[i]=ozlist[i]+dis_z


##plot#####
###
fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot(oxlist, oylist, ozlist, "r-")
ax.plot(xxglist, yyglist, zzglist, "b-")
plt.show()

# gt_data_list_100 = gt_data_list[0:100]


# take output_list 2135
# take gt_data_list 22402
# go through output_list item each
# find gt_data corresponding by t value
# make list of [t gx gy gz gq1 gq2 gq3 gq4 ox oy oz oq1 oq2 oq3 oq4]

####SYNSH LIST FUNCTION



######


###real match
'''

def return_gt_data (gt_data_list, out_data):
    # compare frame info item 0 and gt_data item 0
    diff = 100000000000
    no = 0
    record = -1
    f_gt_data = gt_data_list[0]
    for gt_data in gt_data_list:
         ot = out_data[0]
         gt = gt_data[0]
         if (diff > abs(ot - gt)):
             diff = abs(ot - gt)
             # print("diff: ", diff)
             record = no
             f_gt_data = gt_data 
             # print("found")
         #else:
         #    print("not found")           
         no = no + 1
    ##print(no)
    ##print(record)
    ##print(diff)
    ##print(out_data)
    ##print(f_gt_data)
    return [f_gt_data[0], f_gt_data[1], f_gt_data[2], f_gt_data[3], f_gt_data[4], f_gt_data[5], f_gt_data[6], f_gt_data[7], out_data[1], out_data[2], out_data[3], out_data[4], out_data[5], out_data[6], out_data[7]]
'''
# testing
# return_gt_data (gt_data_list, output_list[100])   




# T1: T inverse to unknown ref is T-1 with respect to first of o 
# T2: T from unknown to first of g
# T = [tx ty tz r1 r2 r3 r4] format

# temp = rotq (0, -refo[1],-refo[2],-refo[3],refo[4],-refo[5],-refo[6],-refo[7]) 
# T1 = [temp[1],temp[2],temp[3],refo[4],-refo[5],-refo[6],-refo[7]]  
# T2 = [refg[1],refg[2],refg[3],refg[4],-refg[5],-refg[6],-refg[7]]  

synch_list = []
'''
for oitem in output_list:
    temp = return_gt_data (gt_data_list, oitem)
    synch_list.append(temp)

'''
# first = synch_list[10]

# refo = [first[0],first[8],first[9],first[10],first[11],first[12],first[13],first[14]]
# refg = [first[0],first[1],first[2],first[3],first[4],first[5],first[6],first[7]]

ggxlist = []
ggylist = []
ggzlist = []
ooxlist = []
ooylist = []
oozlist = []

# dx = refg[1] - refo[1]
# dy = refg[2] - refo[2]
# dz = refg[3] - refo[3]

for i in synch_list:
    ##print(i[0])
    ggxlist.append(i[1])
    ggylist.append(i[2])
    ggzlist.append(i[3])
    ooxlist.append(i[8])
    ooylist.append(i[9])
    oozlist.append([10])





#fig = plt.figure(figsize=(10, 5))

#ax = fig.add_subplot(111, projection='3d') # Axe3D object

#ax.plot(ggxlist, ggylist, ggzlist, "b-")
#plt.show()


def best_fit_transform(A, B):

#  
#    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
#    Input:
#      A: Nxm numpy array of corresponding points
#      B: Nxm numpy array of corresponding points
#    Returns:
#      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
#      R: mxm rotation matrix
#      t: mx1 translation vector
#    
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)
    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t



# Constants
N = 10                                    # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])



def test_best_fit():
    # Generate a random dataset
    A = np.random.rand(N, dim)
    total_time = 0
    for i in range(num_tests):
        B = np.copy(A)
        # Translate
        t = np.random.rand(dim)*translation
        B += t
        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T
        # Add noise
        B += np.random.randn(N, dim) * noise_sigma
        # Find best fit transform
        start = time.time()
        T, R1, t1 = best_fit_transform(B, A)
        total_time += time.time() - start
        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B
        # Transform C
        C = np.dot(T, C.T).T
        #assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        #assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        #assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses
    #print('best fit time: {:.3}'.format(total_time/num_tests))
    #print(t, R)
    #print(A)
    #print(B)
    return

# construct the numpy array with ggxlist ggylist ggzlist ooxlist ooylist oozlist
# or the synch_list
GL = []
OL = []

print(len(xxglist))
print(len(oxlist))

for i in range(len(xxglist)):
    GL.append([xxglist[i], yyglist[i], zzglist[i]])  
    OL.append([oxlist[i], oylist[i], ozlist[i]])

GN = np.asarray(GL)
ON = np.asarray(OL)

 

def test_best_fit2(A, B, nu_tests):
    total_time = 0
    for i in range(nu_tests):
        # Find best fit transform
        start = time.time()
        T, R1, t1 = best_fit_transform(B, A)
        total_time += time.time() - start
        # Make C a homogeneous representation of B
        N = len(xxglist)
        C = np.ones((N, 4))
        C[:,0:3] = B
        # Transform C
        C = np.dot(T, C.T).T
        #assert np.allclose(C[:,0:3], A, atol=noise_sigma) # T should transform B (or C) to A
    #print('best fit time: {:.3}'.format(total_time/num_tests))
    #print(t1, R1)
    #print(B[10])
    #print(A[10])
    #print(C[10])
    #print(B[100])
    #print(A[100])
    #print(C[100])
    return C

final = test_best_fit2(GN, ON, 10)



ggxlist = []
ggylist = []
ggzlist = []

gqwlist = []
gqxlist = []
gqylist = []
gqzlist = []



ooxlist = []
ooylist = []
oozlist = []

oqwlist = []
oqxlist = []
oqylist = []
oqzlist = []


xaelist = []
yaelist = []
zaelist = []



for i in GN:
    ggxlist.append(i[0])
    ggylist.append(i[1])
    ggzlist.append(i[2])
    gqwlist.append(i[3])
    gqxlist.append(i[4])
    gqylist.append(i[5])
    gqzlist.append(i[6])
       
    
    

for i in final:
    ooxlist.append(i[0])
    ooylist.append(i[1])
    oozlist.append(i[2])
    oqwlist.append(i[3])
    oqxlist.append(i[4])
    oqylist.append(i[5])
    oqzlist.append(i[6])




print(len(ggxlist))
print(len(ooxlist))



#####################################################################################
'''
gx=np.array(ggxlist)
gy=np.array(ggylist)
gz=np.array(ggzlist)

ox=np.array(ooxlist)
oy=np.array(ooylist)
oz=np.array(oozlist)


np.savetxt('C:/Users/HANBIN/Desktop/MAXST관련/gx.txt',gx)
np.savetxt('C:/Users/HANBIN/Desktop/MAXST관련/gy.txt',gy)
np.savetxt('C:/Users/HANBIN/Desktop/MAXST관련/gz.txt',gz)

np.savetxt('C:/Users/HANBIN/Desktop/MAXST관련/ox.txt',ox)
np.savetxt('C:/Users/HANBIN/Desktop/MAXST관련/oy.txt',oy)
np.savetxt('C:/Users/HANBIN/Desktop/MAXST관련/oz.txt',oz)


'''
sum_ae=0

print(len(ggxlist))
for i in range(len(ggxlist)):
    x_dis = abs(ggxlist[i] - ooxlist[i])
    y_dis = abs(ggylist[i] - ooylist[i])
    z_dis = abs(ggzlist[i] - oozlist[i])    
    sum_ae += math.sqrt((x_dis*x_dis) + (y_dis*y_dis) + (z_dis*z_dis))

print("===========================================")
print("Absolute Position Error(m):")
APE = sum_ae/len(ggxlist)
APE = math.sqrt(APE)

print(APE)


#Relative Error

sum_re=0

for i in range(len(ggxlist)-1):
    
    xe=(ooxlist[i+1]-ooxlist[i])**2
    ye=(ooylist[i+1]-ooylist[i])**2
    ze=(oozlist[i+1]-oozlist[i])**2
    alle = math.sqrt(xe+ye+ze)#|pslam(i+1) - pslam(i)|
    
    xg=(ggxlist[i+1]-ggxlist[i])**2
    yg=(ggylist[i+1]-ggylist[i])**2
    zg=(ggzlist[i+1]-ggzlist[i])**2
    
    allg = math.sqrt(xg+yg+zg)#|pground(i+1) - pground(i)|

    sum_re+= (alle-allg)**2

print("===========================================")
print("Relative Position Error(m):")
#print(sum_re)

RPE = sum_re/(len(ggxlist)-1)
RPE = math.sqrt(RPE)
#print(sum_re/(len(ggxlist)-1))
print(RPE)


# 3d

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot([ooxlist[10]], [ooylist[10]], [oozlist[10]], "ro")
ax.plot([ggxlist[10]], [ggylist[10]], [ggzlist[10]], "bo")
ax.plot(ooxlist, ooylist, oozlist, "r-")#estimate
ax.plot(ggxlist, ggylist, ggzlist, "g-")
plt.show()



# 2d
fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot(ooxlist, ooylist, "r-")
ax.plot(ggxlist, ggylist, "black")
plt.show()

#2d-2
fig = plt.figure(figsize=(10, 10))

plt.plot(ooxlist,ooylist,"r-")
plt.xlabel('x [m]')
plt.plot(ggxlist,ggylist,"black")
plt.ylabel('y [m]')
plt.gca().set_aspect("equal")
plt.show()

