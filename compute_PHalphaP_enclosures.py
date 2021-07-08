# This script forms and diagonalizes Q^{\alpha,\perp} P_{\Xi} H^\alpha P_{\Xi} Q^{\alpha,\perp} along a grid of \alpha values while storing the values of various coefficients which are important in the computer-assisted proof of Proposition IV.5

# code for forming and diagonalizing P H^alpha P along a grid of alpha values

import numpy as np
from numpy import linalg as LA 
from scipy import linalg as SLA
import matplotlib.pyplot as plt 
from matplotlib import rc
import numpy.polynomial.polynomial as poly

# parameters of alpha grid

N_alphas = 300000 #300000 is enough!
alpha_0 = 0
alpha_1 = 0.7 

# form grid of alpha values

alpha_grid = np.linspace(alpha_0,alpha_1,N_alphas)
h = alpha_grid[1] - alpha_grid[0]
h_must_be_smaller_than = (388831)**(-1)
print("alpha grid spacing: "+str(h))
print("alpha grid spacing should be smaller than: "+str(h_must_be_smaller_than))
#if h > h_must_be_smaller_than:
    #quit()

# constants

phi = 2*np.pi/3
q1 = -1j
b1 = (1/2)*(np.sqrt(3) + 3*1j)
b2 = (1/2)*(-np.sqrt(3) + 3*1j)

# useful functions

def zhat(z):
    return np.exp(1j*np.angle(z))

def zhat_conj(z):
    return np.exp(-1j*np.angle(z))

# generate key, writing vectors in the form a*q1+b*b1+c*b2

def vec_to_word(v):
    # v is the vector a*q1 + b*b1 + c*b2 in the form of an array (a,b,c)
    # output is "aq1_bb1_cb2"
    v_word = str(v[0])+"q1_"+str(v[1])+"b1_"+str(v[2])+"b2"
    return v_word

def word_to_vec(v_word):
    v_word_split = v_word.split('_')
    v_word_split[0] = v_word_split[0].replace('q1','')
    v_word_split[1] = v_word_split[1].replace('b1','')
    v_word_split[2] = v_word_split[2].replace('b2','')
    v = np.array( [int(v_word_split[0]),int(v_word_split[1]),int(v_word_split[2])] , dtype="int" )
    return v

def vec_to_coords(v):
    coords = v[0]*q1 + v[1]*b1 + v[2]*b2
    return coords

def R_phi_star(v):
    # v is the vector a*q1 + b*b1 + c*b2 in the form of an array (a,b,c)
    R_phi_star = np.array( [[ 1, 0, 0 ],[ 0, 0, 1 ],[ 1, -1, -1 ]] , dtype="int" )
    # result is another vector (a',b',c') such that R_phi_star( a*q1 + b*b1 + c*b2 ) = a'*q1 + b'*b1 + c'*b2
    return R_phi_star@v

# here put vectors to be included in the basis, first entry has to be q1 otherwise code below will break
# eval 1
seeds = [ np.array([1,0,0]) ]
# eval sqrt(3)
seeds.extend( [ np.array([0,-1,0]) , np.array([0,0,-1]) ] )
# eval 2
seeds.extend( [ np.array([1,1,1]) ] )
# eval sqrt(7)
seeds.extend( [ np.array([1,-1,0]) , np.array([1,0,-1]) ] )
# eval 3
seeds.extend( [ np.array([0,1,1]) , np.array([0,-1,-1]) ] )
# eval 2 sqrt(3)
seeds.extend( [ np.array([0,-2,0]) , np.array([0,0,-2]) ] )
# eval sqrt(13)
seeds.extend( [ np.array([1,1,-2]) , np.array([1,-2,1]) ] )
# eval 4
seeds.extend( [ np.array([1,-1,-1]) ] )
# eval sqrt(19)
seeds.extend( [ np.array([1,-2,0]) , np.array([1,0,-2]) ] )
# eval sqrt(21)
seeds.extend( [ np.array([0,-1,-2]) , np.array([0,-2,-1]) , np.array([0,1,2]) , np.array([0,2,1]) ] )
# eval 5
seeds.extend( [ np.array([1,2,2]) ] )
# eval 3 sqrt(3)
seeds.extend( [ np.array([0,-3,0]) , np.array([0,0,-3]) ] )
# eval 2 sqrt(7)
seeds.extend( [ np.array([1,1,3]) , np.array([1,3,1]) ] )
# eval sqrt(31)
seeds.extend( [ np.array([1,-2,-1]) , np.array([1,-1,-2]) ] )
# eval 6
seeds.extend( [ np.array([0,2,2]) , np.array([0,-2,-2]) ] )
# eval sqrt(37)
seeds.extend( [ np.array([1,0,4]) , np.array([1,4,0]) ] )
# eval sqrt(39)
seeds.extend( [ np.array([0,1,3]) , np.array([0,3,1]) , np.array([0,-1,4]) , np.array([0,4,-1]) ] )
# eval sqrt(43)
seeds.extend( [ np.array([1,2,3]) , np.array([1,3,2]) ] )
# eval 4 sqrt(3)
seeds.extend( [ np.array([0,0,4]) , np.array([0,4,0]) ] )
# eval 7 (only two! there is a third which we ignore to get $P H^1 P^\perp \leq 1$)
seeds.extend( [ np.array([1,-4,1]) , np.array([1,1,-4]) ] )

# initiate key with 0 entry
key = { "0q1_0b1_0b2":len(seeds) } #0

# fill in rest of key
for m in range(len(seeds)): 
    key[vec_to_word(seeds[m])+"_-1"] = m #m+1
    key[vec_to_word(R_phi_star(seeds[m]))+"_-1"] = m #m+1
    key[vec_to_word(R_phi_star(R_phi_star(seeds[m])))+"_-1"] = m #m+1
    key[vec_to_word(seeds[m])+"_+1"] = m+1+len(seeds) 
    key[vec_to_word(R_phi_star(seeds[m]))+"_+1"] = m+1+len(seeds)
    key[vec_to_word(R_phi_star(R_phi_star(seeds[m])))+"_+1"] = m+1+len(seeds)

modes = 1 + max(key.values()) 

def e(n):
    e = np.zeros( modes , dtype="complex" )
    e[n] = 1
    return e

def PH0P():

    PH0P = np.zeros( (modes,modes) , dtype="complex" )

    # add P H0 P entries

    for seed in seeds: 
        length = np.abs(vec_to_coords(seed))
        PH0P[:,key[vec_to_word(seed)+"_+1"]] += length*e(key[vec_to_word(seed)+"_-1"])

    PH0P += np.conjugate(np.transpose(PH0P))

    return PH0P

def PH1P():

    PH1P = np.zeros( (modes,modes) , dtype="complex" )

    # add P H1 P entries

    PH1P[:,key["0q1_0b1_0b2"]] += np.sqrt(3)*zhat_conj(q1)*e(key["1q1_0b1_0b2_-1"])

    for seed in seeds:
        if seed[0] == 1 and seed[1] == 0 and seed[2] == 0:
            if "0q1_-1b1_0b2_-1" in key:
                PH1P[:,key["1q1_0b1_0b2_+1"]] += np.exp(1j*phi)*zhat_conj(-b1)*e(key["0q1_-1b1_0b2_-1"])
            if "0q1_0b1_-1b2_-1" in key:
                PH1P[:,key["1q1_0b1_0b2_+1"]] += np.exp(-1j*phi)*zhat_conj(-b2)*e(key["0q1_0b1_-1b2_-1"])
        if seed[0] == 0 and ( seed[1] != 0 or seed[2] != 0 ):
            if vec_to_word(seed+np.array([1,0,0]))+"_-1" in key:
                PH1P[:,key[vec_to_word(seed)+"_+1"]] += zhat_conj(q1+vec_to_coords(seed))*e(key[vec_to_word(seed+np.array([1,0,0]))+"_-1"])
            if vec_to_word(seed+np.array([1,1,0]))+"_-1" in key:
                PH1P[:,key[vec_to_word(seed)+"_+1"]] += np.exp(1j*phi)*zhat_conj(q1+b1+vec_to_coords(seed))*e(key[vec_to_word(seed+np.array([1,1,0]))+"_-1"])
            if vec_to_word(seed+np.array([1,0,1]))+"_-1" in key:
                PH1P[:,key[vec_to_word(seed)+"_+1"]] += np.exp(-1j*phi)*zhat_conj(q1+b2+vec_to_coords(seed))*e(key[vec_to_word(seed+np.array([1,0,1]))+"_-1"])
        if seed[0] == 1 and ( seed[1] != 0 or seed[2] != 0 ):
            if vec_to_word(seed+np.array([-1,0,0]))+"_-1" in key:
                PH1P[:,key[vec_to_word(seed)+"_+1"]] += zhat_conj(-q1+vec_to_coords(seed))*e(key[vec_to_word(seed+np.array([-1,0,0]))+"_-1"])
            if vec_to_word(seed+np.array([-1,-1,0]))+"_-1" in key:
                PH1P[:,key[vec_to_word(seed)+"_+1"]] += np.exp(1j*phi)*zhat_conj(-q1-b1+vec_to_coords(seed))*e(key[vec_to_word(seed+np.array([-1,-1,0]))+"_-1"])
            if vec_to_word(seed+np.array([-1,0,-1]))+"_-1" in key:
                PH1P[:,key[vec_to_word(seed)+"_+1"]] += np.exp(-1j*phi)*zhat_conj(-q1-b2+vec_to_coords(seed))*e(key[vec_to_word(seed+np.array([-1,0,-1]))+"_-1"])

    PH1P += np.conjugate(np.transpose(PH1P))

    return PH1P

def PHalphaP(alpha):
    return PH0P() + alpha*PH1P()

def zero_mode_0():
    return e(key["0q1_0b1_0b2"]) 

def zero_mode_1(alpha):
    return zero_mode_0() + alpha*( - np.sqrt(3)*1j*e(key["1q1_0b1_0b2_+1"]) )

def zero_mode_2(alpha):
    return zero_mode_1(alpha) + alpha**2*( 1j*np.exp(-1j*phi)*e(key["0q1_-1b1_0b2_+1"]) - 1j*np.exp(1j*phi)*e(key["0q1_0b1_-1b2_+1"]) )

def zero_mode_3(alpha):
    return zero_mode_2(alpha) + alpha**3*( 1j*np.exp(1j*phi)*zhat_conj(q1-b2)*np.sqrt(7)**(-1)*e(key["1q1_0b1_-1b2_+1"]) - 1j*np.exp(-1j*phi)*zhat_conj(q1-b1)*np.sqrt(7)**(-1)*e(key["1q1_-1b1_0b2_+1"]) )

def zero_mode_4(alpha):
    return zero_mode_3(alpha) + alpha**4*( -1j*np.exp(1j*phi)*zhat_conj(q1-b2)*np.sqrt(7)**(-1)*( zhat_conj(-b2)*np.sqrt(3)**(-1)*e(key["0q1_0b1_-1b2_+1"]) + np.exp(-1j*phi)*zhat_conj(-2*b2)*(2*np.sqrt(3))**(-1)*e(key["0q1_0b1_-2b2_+1"]) ) + 1j*np.exp(-1j*phi)*zhat_conj(q1-b1)*np.sqrt(7)**(-1)*( zhat_conj(-b1)*np.sqrt(3)**(-1)*e(key["0q1_-1b1_0b2_+1"]) + np.exp(1j*phi)*zhat_conj(-2*b1)*(2*np.sqrt(3))**(-1)*e(key["0q1_-2b1_0b2_+1"]) ) + 1j*zhat_conj(-b1-b2)*(3*np.sqrt(7))**(-1)*( np.exp(1j*phi)*zhat_conj(q1-b1) - np.exp(-1j*phi)*zhat_conj(q1-b2) )*e(key["0q1_-1b1_-1b2_+1"]) )

def zero_mode_5(alpha):
    return zero_mode_4(alpha) + alpha**5*( 
    1j*np.exp(1j*phi)*zhat_conj(q1-b2)*np.sqrt(7)**(-1)*(
    zhat_conj(-b2)*np.sqrt(3)**(-1)*(
    zhat_conj(q1-b2)*np.sqrt(7)**(-1)*e(key["1q1_0b1_-1b2_+1"])
    + np.exp(1j*phi)*zhat_conj(q1+b1-b2)*2**(-1)*e(key["1q1_1b1_-1b2_+1"])
    + np.exp(-1j*phi)*zhat_conj(q1)*e(key["1q1_0b1_0b2_+1"])
    )
    + np.exp(1j*phi)*zhat_conj(-b2-b1)*3**(-1)*(
    zhat_conj(q1-b2-b1)*4**(-1)*e(key["1q1_-1b1_-1b2_+1"])
    + np.exp(1j*phi)*zhat_conj(q1-b2)*np.sqrt(7)**(-1)*e(key["1q1_0b1_-1b2_+1"])
    + np.exp(-1j*phi)*zhat_conj(q1-b1)*np.sqrt(7)**(-1)*e(key["1q1_-1b1_0b2_+1"])
    )
    + np.exp(-1j*phi)*zhat_conj(-2*b2)*(2*np.sqrt(3))**(-1)*(
    zhat_conj(q1-2*b2)*np.sqrt(19)**(-1)*e(key["1q1_0b1_-2b2_+1"])
    + np.exp(1j*phi)*zhat_conj(q1 + b1 - 2*b2)*np.sqrt(13)**(-1)*e(key["1q1_1b1_-2b2_+1"])
    + np.exp(-1j*phi)*zhat_conj(q1 - b2)*np.sqrt(7)**(-1)*e(key["1q1_0b1_-1b2_+1"])
    )
    )
    - 1j*np.exp(-1j*phi)*zhat_conj(q1-b1)*np.sqrt(7)**(-1)*(
    zhat_conj(-b1)*np.sqrt(3)**(-1)*(
    zhat_conj(q1-b1)*np.sqrt(7)**(-1)*e(key["1q1_-1b1_0b2_+1"])
    + np.exp(1j*phi)*zhat_conj(q1)*e(key["1q1_0b1_0b2_+1"])
    + np.exp(-1j*phi)*zhat_conj(q1+b2-b1)*2**(-1)*e(key["1q1_-1b1_1b2_+1"])
    )
    + np.exp(1j*phi)*zhat_conj(-2*b1)*(2*np.sqrt(3))**(-1)*(
    zhat_conj(q1-2*b1)*np.sqrt(19)**(-1)*e(key["1q1_-2b1_0b2_+1"])
    + np.exp(1j*phi)*zhat_conj(q1-b1)*np.sqrt(7)**(-1)*e(key["1q1_-1b1_0b2_+1"])
    + np.exp(-1j*phi)*zhat_conj(q1+b2-2*b1)*np.sqrt(13)**(-1)*e(key["1q1_-2b1_1b2_+1"])
    )
    + np.exp(-1j*phi)*zhat_conj(-b1-b2)*3**(-1)*(
    zhat_conj(q1-b1-b2)*4**(-1)*e(key["1q1_-1b1_-1b2_+1"])
    + np.exp(1j*phi)*zhat_conj(q1-b2)*np.sqrt(7)**(-1)*e(key["1q1_0b1_-1b2_+1"])
    + np.exp(-1j*phi)*zhat_conj(q1-b1)*np.sqrt(7)**(-1)*e(key["1q1_-1b1_0b2_+1"])
    )
    )
    )

def zero_mode_6(alpha):
    return zero_mode_5(alpha) + alpha**6*( 
    ( np.sqrt(91)/42 )*( (9*np.sqrt(273) - 11*np.sqrt(91)*1j)/182 )*e(key["0q1_-1b1_0b2_+1"])
    + ( 4*np.sqrt(1729)/5187 )*( (-45*np.sqrt(5187) - 29*np.sqrt(1729)*1j)/3458 )*e(key["0q1_-2b1_0b2_+1"])
    + ( np.sqrt(91)/42 )*( (9*np.sqrt(273) + 11*np.sqrt(91)*1j)/182 )*e(key["0q1_0b1_-1b2_+1"])
    - ( np.sqrt(3)/26 )*e(key["0q1_-2b1_1b2_+1"])  
    + ( np.sqrt(133)/2394 )*( (9*np.sqrt(399) - 17*np.sqrt(133)*1j)/266 )*e(key["0q1_-3b1_0b2_+1"])
    + ( np.sqrt(57)/798 )*( (59*np.sqrt(19) - 9*np.sqrt(57)*1j)/266 )*e(key["0q1_-2b1_-1b2_+1"])
    + ( np.sqrt(13)/546 )*( (-17*np.sqrt(39) - 41*np.sqrt(13)*1j)/182 )*e(key["0q1_-3b1_1b2_+1"])
    + ( np.sqrt(57)/798 )*( (59*np.sqrt(19) + 9*np.sqrt(57)*1j)/266 )*e(key["0q1_-1b1_-2b2_+1"])
    + ( 4*np.sqrt(1729)/5187 )*( (-45*np.sqrt(5187) + 29*np.sqrt(1729)*1j)/3458 )*e(key["0q1_0b1_-2b2_+1"])
    + ( np.sqrt(133)/2394 )*( (9*np.sqrt(399) + 17*np.sqrt(133)*1j)/266 )*e(key["0q1_0b1_-3b2_+1"])
    + ( np.sqrt(13)/546 )*( (-17*np.sqrt(39) + 41*np.sqrt(13)*1j)/182 )*e(key["0q1_1b1_-3b2_+1"])
    )

def zero_mode_7(alpha):
    return zero_mode_6(alpha) + alpha**7*(
    ( np.sqrt(1032213)/10374 )*( (-97*np.sqrt(1032213) - 562*np.sqrt(344071)*1j)/344071 )*e(key["1q1_-1b1_0b2_+1"])
    - (np.sqrt(3)*1j/42)*e(key["1q1_0b1_0b2_+1"]) - (2*np.sqrt(3)*1j/273)*e(key["1q1_-1b1_1b2_+1"])
    + ( np.sqrt(3549637)/217854 )*( (-2621*np.sqrt(3549637) + 1563*np.sqrt(10648911)*1j)/7099274 )*e(key["1q1_-2b1_0b2_+1"])
    + ( np.sqrt(178087)/24206 )*( (-241*np.sqrt(178087) + 467*np.sqrt(534261)*1j)/356174 )*e(key["1q1_-2b1_1b2_+1"])
    + ( np.sqrt(1032213)/10374 )*( (97*np.sqrt(1032213) - 562*np.sqrt(344071)*1j)/344071 )*e(key["1q1_0b1_-1b2_+1"])
    + ( np.sqrt(178087)/24206 )*( (241*np.sqrt(178087) + 467*np.sqrt(534261)*1j)/356174 )*e(key["1q1_-2b1_2b2_+1"])
    + ( np.sqrt(4921)/88578 )*( (-53*np.sqrt(4921) - 75*np.sqrt(14763)*1j)/9842 )*e(key["1q1_-3b1_0b2_+1"])
    + ( 2*np.sqrt(247)/15561 )*( (-215*np.sqrt(247) + 27*np.sqrt(741)*1j)/3458 )*e(key["1q1_-3b1_1b2_+1"])
    + ( np.sqrt(1767)/24738 )*( (-10*np.sqrt(1767) - 169*np.sqrt(589)*1j)/4123 )*e(key["1q1_-2b1_-1b2_+1"])
    + ( 2*np.sqrt(3)*1j/2793 )*e(key["1q1_-1b1_-1b2_+1"])
    + ( 29*np.sqrt(3)*1j/19110 )*e(key["1q1_-3b1_2b2_+1"])
    + ( np.sqrt(1767)/24738 )*( (10*np.sqrt(1767) - 169*np.sqrt(589)*1j)/4123 )*e(key["1q1_-1b1_-2b2_+1"])
    + ( np.sqrt(3549637)/217854 )*( (2621*np.sqrt(3549637) + 1563*np.sqrt(10648911)*1j)/7099274 )*e(key["1q1_0b1_-2b2_+1"])
    + ( np.sqrt(4921)/88578 )*( (53*np.sqrt(4921) - 75*np.sqrt(14763)*1j)/9842 )*e(key["1q1_0b1_-3b2_+1"])
    + ( 2*np.sqrt(247)/15561 )*( (215*np.sqrt(247) + 27*np.sqrt(741)*1j)/3458 )*e(key["1q1_1b1_-3b2_+1"])
    )

def inner(u,v):
    # returns the inner product <u,v> with complex conjugation
    return np.inner(np.conjugate(u),v)

def normalize(v):
    # normalizes a vector
    return v*( np.sqrt(inner(v,v)) )**(-1)

def zero_mode_8(alpha):
    unnormalized_mode = zero_mode_7(alpha) + alpha**8*(
    ( np.sqrt(160797)/10374 )*( (-206*np.sqrt(53599) - 61*np.sqrt(160797)*1j)/53599 )*e(key["0q1_-1b1_0b2_+1"])
    + ( np.sqrt(1694251299)/1307124 )*( (16249*np.sqrt(564750433) - 10012*np.sqrt(1694251299)*1j)/564750433 )*e(key["0q1_-2b1_0b2_+1"])
    + ( 317*np.sqrt(3)/11466 )*e(key["0q1_-1b1_-1b2_+1"]) 
    + ( np.sqrt(160797)/10374 )*( (-206*np.sqrt(53599) + 61*np.sqrt(160797)*1j)/53599 )*e(key["0q1_0b1_-1b2_+1"])
    + ( 67*np.sqrt(3)/16758 )*e(key["0q1_-2b1_1b2_+1"])
    + ( np.sqrt(837273)/620046 )*( (-496*np.sqrt(279091) - 105*np.sqrt(837273)*1j)/279091 )*e(key["0q1_-3b1_0b2_+1"])
    + ( np.sqrt(997694607)/20260422 )*( (5849*np.sqrt(332564869) - 20785*np.sqrt(997694607)*1j)/665129738 )*e(key["0q1_-2b1_-1b2_+1"])
    + ( np.sqrt(2667)/13230 )*( (-59*np.sqrt(889) - 5*np.sqrt(2667)*1j)/1778 )*e(key["0q1_-3b1_1b2_+1"])
    + ( np.sqrt(1694251299)/1307124 )*( (16249*np.sqrt(564750433) + 10012*np.sqrt(1694251299)*1j)/564750433 )*e(key["0q1_0b1_-2b2_+1"])
    + ( np.sqrt(2667)/13230 )*( (-59*np.sqrt(889) + 5*np.sqrt(2667)*1j)/1778 )*e(key["0q1_-3b1_2b2_+1"])
    + ( np.sqrt(14763)/1062936 )*( (43*np.sqrt(4921) - 32*np.sqrt(14763)*1j)/4921 )*e(key["0q1_-4b1_0b2_+1"])
    + ( np.sqrt(114919077)/39454506 )*( (11413*np.sqrt(38306359) - 2767*np.sqrt(114919077)*1j)/76612718 )*e(key["0q1_-3b1_-1b2_+1"])
    + ( 2*np.sqrt(57)/46683 )*( (-29*np.sqrt(19) - 31*np.sqrt(57)*1j)/266 )*e(key["0q1_-4b1_1b2_+1"])
    + ( 199*np.sqrt(3)/1038996 )*e(key["0q1_-2b1_-2b2_+1"])
    + ( np.sqrt(997694607)/20260422 )*( (5849*np.sqrt(332564869) + 20785*np.sqrt(997694607)*1j)/665129738 )*e(key["0q1_-1b1_-2b2_+1"])
    - ( 29*np.sqrt(3)/114660 )*e(key["0q1_-4b1_2b2_+1"])
    + ( np.sqrt(114919077)/39454506 )*( (11413*np.sqrt(38306359) + 2767*np.sqrt(114919077)*1j)/76612718 )*e(key["0q1_-1b1_-3b2_+1"])
    + ( np.sqrt(837273)/620046 )*( (-496*np.sqrt(279091) + 105*np.sqrt(837273)*1j)/279091 )*e(key["0q1_0b1_-3b2_+1"])
    + ( np.sqrt(14763)/1062936 )*( (43*np.sqrt(4921) + 32*np.sqrt(14763)*1j)/4921 )*e(key["0q1_0b1_-4b2_+1"])
    + ( 2*np.sqrt(57)/46683 )*( (-29*np.sqrt(19) + 31*np.sqrt(57)*1j)/266 )*e(key["0q1_1b1_-4b2_+1"])
    )
    return normalize(unnormalized_mode)

def trunc_PHalphaP(alpha):
    Q = np.outer(zero_mode_8(alpha),np.conjugate(zero_mode_8(alpha)))
    Qperp = np.eye(modes) - Q
    trunc_PHalphaP = Qperp@PHalphaP(alpha)@Qperp
    return trunc_PHalphaP

def norm(v):
    return np.sqrt(inner(v,v))

# get machine epsilon
eps = np.finfo(np.complex128).eps

print("Machine epsilon: " + str(eps))

# if modes = 81, this gives 40. For extracting the positive eigenvalues 
positive_modes = (modes - 1)//2

# numbers to keep track of
smallest_positive_eigenvalues = np.zeros( (N_alphas) , dtype="complex" )
largest_gramian_minus_I_entry = np.zeros( (N_alphas) , dtype="complex" )
largest_entry_of_eigenvectors = np.zeros( (N_alphas) , dtype="complex" )
largest_entry_of_residuals = np.zeros( (N_alphas) , dtype="complex" )
largest_entry_of_H = np.zeros( (N_alphas) , dtype="complex" )
largest_eigenvalue = np.zeros( (N_alphas) , dtype="complex" )

for alpha_idx in range(N_alphas):
    # get alpha value
    alpha = alpha_grid[alpha_idx]
    # form matrix
    matrix = trunc_PHalphaP(alpha)
    # fill in largest entry of H
    largest_entry_of_H[alpha_idx] = np.amax(np.abs(matrix))
    # diagonalize matrix
    evals, evecs = LA.eigh(matrix)
    # fill in smallest positive eigenvalue, note that symmetry of the spectrum means it isn't necessary to search through the eigenvalues (it could happen that the first non-zero positive eigenvalue becomes zero at some point, in which case this entry would become zero) 
    smallest_positive_eigenvalues[alpha_idx] = evals[-40]
    # fill in largest eigenvalue
    largest_eigenvalue[alpha_idx] = np.amax(evals)
    # fill in largest entry of eigenvectors
    largest_entry_of_eigenvectors[alpha_idx] = np.amax(np.abs(evecs))
    # form the gramian matrix of the eigenvectors
    gramian = np.zeros( (modes,modes) , dtype="complex" )
    for i in range(modes):
        for j in range(modes):
            gramian[i,j] = inner(evecs[:,i],evecs[:,j])
    # find largest gramian minus I entry
    largest_gramian_minus_I_entry[alpha_idx] = np.amax(np.abs(gramian - np.eye(modes)))
    # find largest residual entry
    residual_matrix = matrix@evecs - np.transpose(np.diag(evals)@np.transpose(evecs))
    largest_entry_of_residuals[alpha_idx] = np.amax(np.abs(residual_matrix))

print("smallest positive eigenvalue: " + str(np.amin(smallest_positive_eigenvalues)))
print("largest Gramian minus I entry: " + str(np.amax(largest_gramian_minus_I_entry)))
print("largest eigenvectors entry: " + str(np.amax(largest_entry_of_eigenvectors)))
print("largest residuals entry: " + str(np.amax(largest_entry_of_residuals)))
print("largest entry of H: " + str(np.amax(largest_entry_of_H)))
print("largest eigenvalue: " + str(np.amax(largest_eigenvalue)))
