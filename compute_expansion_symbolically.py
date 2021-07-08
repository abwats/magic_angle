# This script calculates
# (1) the formal expansion of psi^alpha up to order 8, and the norms of each of the terms in the expansion (symbolically)
# (2) the formal expansion of the numerator and denominator of v(alpha) using the formal expansion of psi^alpha (symbolically)
# (3) the L2 norm of H1 Psi^8 (symbolically)
# (4) calculates the roots of the numerator of the Fermi velocity (evaluated in floating point arithmetic from the symbolic formula) with and without error terms (polynomial root-finding)
# (5) evaluates the truncated expansion of the Fermi velocity with worst case error terms (evaluated in floating point arithmetic from the symbolic formula) at alpha = .646
# (6) generates Fig II.1 for the paper 

import numpy as np
from numpy.polynomial import polynomial as P
from sympy import Abs, I, sqrt, re, im, Rational, conjugate, simplify, pi, exp, together, Symbol, N
from sympy.solvers import solve
import matplotlib.pyplot as plt

# define parameters 

# if = N, last term is psi^N 
expansion_order = 8 # in the paper the figures are generated with 8, 40 is possible with a few hours of processing time on my laptop
# gap of PHP
g = Rational(3,4)
# plotting limits
a, b = 0, .69
# number of points
gridpoints = 100

# define array representations of q1, b1, and b2

q1_arr = np.array([1,0,0])
b1_arr = np.array([0,1,0])
b2_arr = np.array([0,0,1])
q2_arr = q1_arr + b1_arr
q3_arr = q1_arr + b2_arr

# define (symbolic) complex number representations of q1, b1, and b2

q1_num = -I
b1_num = Rational(1,2)*(sqrt(3)+3*I)
b2_num = Rational(1,2)*(-sqrt(3)+3*I)
q2_num = q1_num + b1_num
q3_num = q1_num + b2_num

# useful complex numbers

eiphi_num = - Rational(1,2) + sqrt(3)*I/2
eminusiphi_num = - Rational(1,2) - sqrt(3)*I/2

# useful functions for manipulating complex numbers

def zhat(z):
    return z/Abs(z)

def zhat_conj(z):
    return conjugate(z/Abs(z))

# functions for converting between array and string representations of vectors

def vec_to_word(v):
    # input is the vector a*q1 + b*b1 + c*b2 in the form of an array of integers np.array( [a,b,c] )
    # output is "aq1_bb1_cb2"
    v_word = str(v[0])+"q1_"+str(v[1])+"b1_"+str(v[2])+"b2"
    return v_word

def word_to_vec(v_word):
    # input is "aq1_bb1_cb2"
    # output is the vector a*q1 + b*b1 + c*b2 in the form of an array of integers np.array( [a,b,c] )
    v_word_split = v_word.split('_')
    v_word_split[0] = v_word_split[0].replace('q1','')
    v_word_split[1] = v_word_split[1].replace('b1','')
    v_word_split[2] = v_word_split[2].replace('b2','')
    v = np.array( [int(v_word_split[0]),int(v_word_split[1]),int(v_word_split[2])] , dtype="int" )
    return v

# functions for converting array to complex vector representation 

def vec_to_complex(v):
    # input is the vector a*q1 + b*b1 + c*b2 in the form of an array of integers np.array( [a,b,c] )
    # output is (symbolic) complex representation of vector
    comp = v[0]*q1_num + v[1]*b1_num + v[2]*b2_num
    return comp 

# helper for main function

def fix_coeff(vec,coeff):
    # integrate wrt H0
    coeff = - Abs(vec_to_complex(vec))**(-1)*coeff
    return ( together(simplify(Abs(coeff))),together(simplify(zhat(coeff))) )

# same function but to compute H1 acting on psi_n, as opposed to psi_n+1 

def fix_coeff_H1(vec,coeff):
    return ( together(simplify(Abs(coeff))),together(simplify(zhat(coeff))) )

# main function, generates new functions from a single basis function

def find_coeffs_tuple(tup):
    # input is a tuple where first element is a string "aq1_bb1_cb2", second and third elements are (symbolic) coefficients in the form of sympy expressions. The second is the absolute value of the coefficient, and the third is the coefficient divided by the absolute value 
    # output is a list of tuples corresponding to the new eigenfunctions and their coefficients
    v_word, norm_coeff, normalized_coeff = tup[0], tup[1], tup[2]
    coeff = tup[1]*tup[2]
    v_vec = word_to_vec(v_word)
    v_comp = vec_to_complex(v_vec)
    if v_vec[0] == 0:
        if v_vec[1] == 0 and v_vec[2] == 0:
            new_vec_1, new_coeff_1 = v_vec + q1_arr, sqrt(3)*I*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff(new_vec_1,new_coeff_1)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1, normalized_new_coeff_1)]
        else:
            new_vec_1, new_coeff_1 = v_vec + q1_arr, zhat_conj(v_comp + q1_num)*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff(new_vec_1,new_coeff_1) 
            new_vec_2, new_coeff_2 = v_vec + q2_arr, eiphi_num*zhat_conj(v_comp + q2_num)*coeff
            norm_new_coeff_2, normalized_new_coeff_2 = fix_coeff(new_vec_2,new_coeff_2)
            new_vec_3, new_coeff_3 = v_vec + q3_arr, eminusiphi_num*zhat_conj(v_comp + q3_num)*coeff
            norm_new_coeff_3, normalized_new_coeff_3 = fix_coeff(new_vec_3,new_coeff_3)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1,normalized_new_coeff_1),(vec_to_word(new_vec_2),norm_new_coeff_2,normalized_new_coeff_2),(vec_to_word(new_vec_3),norm_new_coeff_3,normalized_new_coeff_3)]
    if v_vec[0] == 1:
        if v_vec[1] == 0 and v_vec[2] == 0:
            new_vec_1, new_coeff_1 = v_vec - q2_arr, eiphi_num*zhat_conj(v_comp - q2_num)*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff(new_vec_1,new_coeff_1)
            new_vec_2, new_coeff_2 = v_vec - q3_arr, eminusiphi_num*zhat_conj(v_comp - q3_num)*coeff
            norm_new_coeff_2, normalized_new_coeff_2 = fix_coeff(new_vec_2,new_coeff_2)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1,normalized_new_coeff_1),(vec_to_word(new_vec_2),norm_new_coeff_2,normalized_new_coeff_2)]
        else:
            new_vec_1, new_coeff_1 = v_vec - q1_arr, zhat_conj(v_comp - q1_num)*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff(new_vec_1,new_coeff_1)
            new_vec_2, new_coeff_2 = v_vec - q2_arr, eiphi_num*zhat_conj(v_comp - q2_num)*coeff
            norm_new_coeff_2, normalized_new_coeff_2 = fix_coeff(new_vec_2,new_coeff_2)
            new_vec_3, new_coeff_3 = v_vec - q3_arr, eminusiphi_num*zhat_conj(v_comp - q3_num)*coeff
            norm_new_coeff_3, normalized_new_coeff_3 = fix_coeff(new_vec_3,new_coeff_3)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1,normalized_new_coeff_1),(vec_to_word(new_vec_2),norm_new_coeff_2,normalized_new_coeff_2),(vec_to_word(new_vec_3),norm_new_coeff_3,normalized_new_coeff_3)]

# same but for H1 instead

def find_coeffs_tuple_H1(tup):
    # input is a tuple where first element is a string "aq1_bb1_cb2", second and third elements are (symbolic) coefficients in the form of sympy expressions. The second is the absolute value of the coefficient, and the third is the coefficient divided by the absolute value 
    # output is a list of tuples corresponding to the new eigenfunctions and their coefficients
    v_word, norm_coeff, normalized_coeff = tup[0], tup[1], tup[2]
    coeff = tup[1]*tup[2]
    v_vec = word_to_vec(v_word)
    v_comp = vec_to_complex(v_vec)
    if v_vec[0] == 0:
        if v_vec[1] == 0 and v_vec[2] == 0:
            new_vec_1, new_coeff_1 = v_vec + q1_arr, sqrt(3)*I*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff_H1(new_vec_1,new_coeff_1)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1, normalized_new_coeff_1)]
        else:
            new_vec_1, new_coeff_1 = v_vec + q1_arr, zhat_conj(v_comp + q1_num)*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff_H1(new_vec_1,new_coeff_1) 
            new_vec_2, new_coeff_2 = v_vec + q2_arr, eiphi_num*zhat_conj(v_comp + q2_num)*coeff
            norm_new_coeff_2, normalized_new_coeff_2 = fix_coeff_H1(new_vec_2,new_coeff_2)
            new_vec_3, new_coeff_3 = v_vec + q3_arr, eminusiphi_num*zhat_conj(v_comp + q3_num)*coeff
            norm_new_coeff_3, normalized_new_coeff_3 = fix_coeff_H1(new_vec_3,new_coeff_3)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1,normalized_new_coeff_1),(vec_to_word(new_vec_2),norm_new_coeff_2,normalized_new_coeff_2),(vec_to_word(new_vec_3),norm_new_coeff_3,normalized_new_coeff_3)]
    if v_vec[0] == 1:
        if v_vec[1] == 0 and v_vec[2] == 0:
            new_vec_1, new_coeff_1 = v_vec - q2_arr, eiphi_num*zhat_conj(v_comp - q2_num)*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff_H1(new_vec_1,new_coeff_1)
            new_vec_2, new_coeff_2 = v_vec - q3_arr, eminusiphi_num*zhat_conj(v_comp - q3_num)*coeff
            norm_new_coeff_2, normalized_new_coeff_2 = fix_coeff_H1(new_vec_2,new_coeff_2)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1,normalized_new_coeff_1),(vec_to_word(new_vec_2),norm_new_coeff_2,normalized_new_coeff_2)]
        else:
            new_vec_1, new_coeff_1 = v_vec - q1_arr, zhat_conj(v_comp - q1_num)*coeff
            norm_new_coeff_1, normalized_new_coeff_1 = fix_coeff_H1(new_vec_1,new_coeff_1)
            new_vec_2, new_coeff_2 = v_vec - q2_arr, eiphi_num*zhat_conj(v_comp - q2_num)*coeff
            norm_new_coeff_2, normalized_new_coeff_2 = fix_coeff_H1(new_vec_2,new_coeff_2)
            new_vec_3, new_coeff_3 = v_vec - q3_arr, eminusiphi_num*zhat_conj(v_comp - q3_num)*coeff
            norm_new_coeff_3, normalized_new_coeff_3 = fix_coeff_H1(new_vec_3,new_coeff_3)
            return [(vec_to_word(new_vec_1),norm_new_coeff_1,normalized_new_coeff_1),(vec_to_word(new_vec_2),norm_new_coeff_2,normalized_new_coeff_2),(vec_to_word(new_vec_3),norm_new_coeff_3,normalized_new_coeff_3)]

# for iterating the previous function over a list  

def find_coeffs_list(lis):
    # input is a list of tuples
    # for each tuple, first element is a string "aq1_bb1_cb2" and second element is a (symbolic) coefficient in the form of a sympy expression
    # output is a list of tuples corresponding to the new eigenfunctions and their coefficients
    new_l = []
    for l in lis:
        new_l = new_l + find_coeffs_tuple(l)
    return new_l

# same but for H1 acting on psi_n instead of for psi_n+1

def find_coeffs_list_H1(lis):
    # input is a list of tuples
    # for each tuple, first element is a string "aq1_bb1_cb2" and second element is a (symbolic) coefficient in the form of a sympy expression
    # output is a list of tuples corresponding to the new eigenfunctions and their coefficients
    new_l = []
    for l in lis:
        new_l = new_l + find_coeffs_tuple_H1(l)
    return new_l

# function which rotates vectors in array representation

def R_phi_star(v):
    # v is the vector a*q1 + b*b1 + c*b2 in the form of an array np.array( [a,b,c] )
    R_phi_star = np.array( [[ 1, 0, 0 ],[ 0, 0, 1 ],[ 1, -1, -1 ]] , dtype="int" )
    # result is another vector np.array( [a',b',c'] ) where a', b', c' are integers such that R_phi_star( a*q1 + b*b1 + c*b2 ) = a'*q1 + b'*b1 + c'*b2
    return R_phi_star@v

# for simplifying a list

def simplify_list(lis):
    # input is a list of tuples
    # goal of this function is to identify tuples which come from the same vector, or rotations of the same vector, and then combine them
    # first, "normalize" so that every representative of an orbit in the list is the same
    num = len(lis)
    for j1 in range(len(lis)): 
        # get representations of vec as arrays from tuple
        vec1 = word_to_vec(lis[j1][0])
        vec2 = R_phi_star(vec1)
        vec3 = R_phi_star(R_phi_star(vec1))
        for j2 in range(j1+1,len(lis)):
            vec4 = word_to_vec(lis[j2][0])
            coeff_1, coeff_2 = lis[j2][1], lis[j2][2]
            if np.array_equal(vec1,vec4):
                lis[j2] = ( vec_to_word(vec1), coeff_1, coeff_2 )
            if np.array_equal(vec2,vec4):
                lis[j2] = ( vec_to_word(vec1), coeff_1, coeff_2 )
            if np.array_equal(vec3,vec4):
                lis[j2] = ( vec_to_word(vec1), coeff_1, coeff_2 )
    # now organize into a new smaller list
    # first get list of vecs and list of vecs without duplicates
    list_of_vecs = []
    for j1 in range(len(lis)):
        list_of_vecs.append(lis[j1][0])
    list_of_vecs_no_duplicates = []
    for entry in list_of_vecs:
        if entry not in list_of_vecs_no_duplicates:
            list_of_vecs_no_duplicates.append(entry)
    # now make the new list of tuples
    lis_2 = []
    for entry in list_of_vecs_no_duplicates:
        vec = entry
        coeff = 0
        for j1 in range(len(lis)):
            if lis[j1][0] == entry:
                coeff += lis[j1][1]*lis[j1][2]
        coeff_1 = together(simplify(Abs(coeff)))
        coeff_2 = together(simplify(zhat(coeff)))
        # only include in new list if the new coefficient is not zero
        if coeff_1 != 0:
            lis_2.append( (vec,coeff_1,coeff_2) )
    # return smaller list
    return lis_2

# calculate the expansion of the Fermi velocity denominator symbolically

# function which takes the inner product of two lists of simplified tuples

def inner(lis_1,lis_2):
    num = 0
    for j1 in range(len(lis_1)):
        # get representations of vec from tuple
        vec1 = lis_1[j1][0]
        vec2 = vec_to_word(R_phi_star(word_to_vec(vec1)))
        vec3 = vec_to_word(R_phi_star(R_phi_star(word_to_vec(vec1))))
        for j2 in range(len(lis_2)):
            if lis_2[j2][0] == vec1 or lis_2[j2][0] == vec2 or lis_2[j2][0] == vec3: 
                num += conjugate(lis_1[j1][1]*lis_1[j1][2])*lis_2[j2][1]*lis_2[j2][2]
    return num

# function which takes the dot product of two lists of simplified tuples

def dot(lis_1,lis_2):
    num = 0
    for j1 in range(len(lis_1)):
        # get representations of vec from tuple
        vec1 = lis_1[j1][0]
        vec2 = vec_to_word(R_phi_star(word_to_vec(vec1)))
        vec3 = vec_to_word(R_phi_star(R_phi_star(word_to_vec(vec1))))
        for j2 in range(len(lis_2)):
            if lis_2[j2][0] == vec1 or lis_2[j2][0] == vec2 or lis_2[j2][0] == vec3: 
                num += lis_1[j1][1]*lis_1[j1][2]*lis_2[j2][1]*lis_2[j2][2]
    return num

# calculate expansion of Fermi velocity numerator

def numerator(psi):
    numerator = []
    # complete first n entries 
    for n in range(expansion_order+1):
        num = 0
        for j in range(n+1):
            num += dot(psi[j],psi[n-j])
        numerator.append( together(simplify(num)) )
    # complete next n entries (note these will likely not be correct in the sense of approximating the exact infinite series)
    for n in reversed(range(expansion_order)):
        num = 0
        for j in range(n+1):
            num += dot(psi[expansion_order-j],psi[expansion_order-(n-j)])
        numerator.append( together(simplify(num)) )
    return numerator

# calculate expansion of Fermi velocity denominator

def denominator(psi):
    denominator = []
    # complete first n entries 
    for n in range(expansion_order+1):
        num = 0
        for j in range(n+1):
            num += inner(psi[j],psi[n-j])
        denominator.append( together(simplify(num)) )
    # complete next n entries (note these will likely not be correct in the sense of approximating the exact infinite series)
    for n in reversed(range(expansion_order)):
        num = 0
        for j in range(n+1):
            num += inner(psi[expansion_order-j],psi[expansion_order-(n-j)])
        denominator.append( together(simplify(num)) )
    return denominator 

# calculate the expansion of psi, and numerator and denominator of v coefficients symbolically

psi = [[("0q1_0b1_0b2",1,1)]]
for j in range(expansion_order):
    psi.append(simplify_list(find_coeffs_list(psi[-1])))

# print terms in formal expansion of psi as a list of pairs: (mode index, coefficient)

print("0th term of formal expansion of psi: "+str(psi[0]))
print("1st term of formal expansion of psi: "+str(psi[1]))
print("2nd term of formal expansion of psi: "+str(psi[2]))
print("3rd term of formal expansion of psi: "+str(psi[3]))
print("4th term of formal expansion of psi: "+str(psi[4]))
print("5th term of formal expansion of psi: "+str(psi[5]))
print("6th term of formal expansion of psi: "+str(psi[6]))
print("7th term of formal expansion of psi: "+str(psi[7]))
print("8th term of formal expansion of psi: "+str(psi[8]))

# calculate the norm of each term
psi_norms = []
for j in range(expansion_order+1):
    psi_norms.append(together(simplify(sqrt(inner(psi[j],psi[j])))))

# print norms of each \Psi^n

print("L2 norms of each \Psi^n: "+str(psi_norms))

numerator_list = numerator(psi)

print("Numerator of Fermi velocity expansion: "+str(numerator_list))

denominator_list = denominator(psi)

print("Denominator of Fermi velocity expansion: "+str(denominator_list))

# calculate the norm of H1 acting on the last term (this is necessary for when we include error terms)

H1_norm_exact = together(simplify(sqrt(inner(find_coeffs_list_H1(psi[-1]),find_coeffs_list_H1(psi[-1])))))

print("L2 norm of H1 Psi^8: "+str(H1_norm_exact))

# use the bound in the paper instead (it is easy to check this number is larger than the previous one)

H1_norm = Rational(3,20)

print("paper bound on H1 Psi^8 minus exact H1 Psi^8: "+str(N(H1_norm - H1_norm_exact)))

# convert numerator_list into list of floats

numerator_list_floats = np.zeros( (len(numerator_list)) , dtype="float" ) 

for j in range(len(numerator_list)):
    numerator_list_floats[j] = N(numerator_list[j])

# compute real and positive roots of the numerator expansion

roots_numerator = P.polyroots(numerator_list_floats)

real_roots_numerator = roots_numerator[np.isreal(roots_numerator)]
positive_real_roots_numerator = real_roots_numerator[real_roots_numerator>0]

print(f"first root of Fermi velocity numerator without error terms: {positive_real_roots_numerator[0].real:.14f}")

# calculate the numerator with the error terms, the polynomial with error terms is
# (g - alpha)^2 ip{ sum alpha^n psi^n }{ sum alpha^n psi^n }
# + 2 (g - alpha) sum alpha^n | psi^n | alpha^{N+1} | H^1 psi^N |
# + alpha^{2 (N+1)} | H^1 psi^N |
# note that (g - alpha)^2 = g^2 - 2 g alpha + alpha^2

# make g (gap of PHP) into a float

g = N(g)

# form first term

g_factor_1 = np.array( [g**2,-2*g,1] , dtype="float" )
first_term = P.polymul( g_factor_1,numerator_list_floats )

# form second term

first_error_term = np.zeros( (expansion_order+1) , dtype="float" )
for n in range(expansion_order+1):
    first_error_term[n] = N(together(simplify(sqrt(inner(psi[n],psi[n]))))*H1_norm)

alpha_Nplus1_factor = np.zeros( (expansion_order+2) , dtype="float" )
alpha_Nplus1_factor[-1] = 1

g_factor_2 = np.array( [2*g,-2] , dtype="float" )

second_term = P.polymul( alpha_Nplus1_factor, P.polymul( first_error_term,g_factor_2 ) )

# form third term 

alpha_2Nplus2_factor = P.polymul( alpha_Nplus1_factor, alpha_Nplus1_factor )
third_term = P.polymul( alpha_2Nplus2_factor , np.array( [N(H1_norm**2)] , dtype="float" ) )

# form new numerator list

numerator_list_floats_with_errors = P.polyadd(first_term,P.polyadd(second_term,third_term))

## find roots of this polynomial

roots_with_errors = P.polyroots(numerator_list_floats_with_errors)

real_roots_with_errors = roots_with_errors[np.isreal(roots_with_errors)]
positive_real_roots_with_errors = real_roots_with_errors[real_roots_with_errors>0]

print(f"first root of Fermi velocity numerator with worst-case error terms: {positive_real_roots_with_errors[0].real:.14f}")

# calculate roots with best case error bound

# form new numerator list

numerator_list_floats_with_errors_bestcase = P.polyadd(first_term,P.polyadd(-second_term,-third_term))

# find roots of this polynomial

roots_bestcase = P.polyroots(numerator_list_floats_with_errors_bestcase)

real_roots_bestcase = roots_bestcase[np.isreal(roots_bestcase)]
positive_real_roots_bestcase = real_roots_bestcase[real_roots_bestcase>0]

print(f"first root of Fermi velocity numerator with best-case error terms: {positive_real_roots_bestcase[0].real:.14f}")

# get error expansions in a form to be plotted

# start with numerator expansion

def numerator(x):
    return P.polyval(x,numerator_list_floats)

# now for numerator with worst-case error terms 

# form first term

first_term = numerator_list_floats

# calculate the norm of H1 acting on the last term

H1_norm = together(simplify(sqrt(inner(find_coeffs_list_H1(psi[-1]),find_coeffs_list_H1(psi[-1])))))

# form second term

first_error_term = np.zeros( (expansion_order+1) , dtype="float" )
for n in range(expansion_order+1):
    first_error_term[n] = N(together(simplify(sqrt(inner(psi[n],psi[n]))))*H1_norm)

alpha_Nplus1_factor = np.zeros( (expansion_order+2) , dtype="float" )
alpha_Nplus1_factor[-1] = 1

g_factor_2 = np.array( [2] , dtype="float" )

second_term = P.polymul( alpha_Nplus1_factor, P.polymul( first_error_term,g_factor_2 ) )

# form third term 

alpha_2Nplus2_factor = P.polymul( alpha_Nplus1_factor, alpha_Nplus1_factor )
third_term = P.polymul( alpha_2Nplus2_factor , np.array( [N(H1_norm**2)] , dtype="float" ) )

def numerator_with_error(x):
    val = P.polyval(x,numerator_list_floats)
    val = val + P.polyval(x,second_term)*(g-x)**(-1)
    val = val + P.polyval(x,third_term)*(g-x)**(-2)
    return val

# now for numerator with best-case error terms 

def numerator_bestcase(x):
    val = P.polyval(x,numerator_list_floats)
    val = val - P.polyval(x,second_term)*(g-x)**(-1)
    val = val - P.polyval(x,third_term)*(g-x)**(-2)
    return val

# check value of numerator with worst-case error terms at specific value of alpha  
test_alpha_1 = .646
print("value of truncated Fermi velocity numerator series with worst-case error at "+str(test_alpha_1)+" is "+str(numerator_with_error(test_alpha_1)))

# to generate figures for paper

X = np.linspace(a,b,gridpoints)

plt.figure()
plt.rc('text', usetex=True)
plt.plot(X,numerator_with_error(X),label=r"8th order expansion of $v(\alpha)$ numerator with worst-case error bound")
plt.plot(X,numerator(X),label=r"8th order expansion of $v(\alpha)$ numerator")
plt.plot(X,numerator_bestcase(X),label=r"8th order expansion of $v(\alpha)$ numerator with best-case error bound")
plt.plot(X,0*X,c='k')
plt.scatter(positive_real_roots_with_errors[0].real,[0])
plt.scatter(positive_real_roots_numerator[0].real,[0])
plt.scatter(positive_real_roots_bestcase[0].real,[0])
plt.xlim([0,.69])
plt.ylim([-2,2])
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$v(\alpha)$")
plt.legend()

plt.figure()
plt.rc('text', usetex=True)
plt.plot(X,numerator_with_error(X))
plt.plot(X,numerator(X))
plt.plot(X,numerator_bestcase(X))
plt.plot(X,0*X,c='k')
plt.scatter(positive_real_roots_with_errors[0].real,[0],label=r"root of expansion of $v(\alpha)$ numerator w/ worst-case error at $\alpha = $"+str(np.around(positive_real_roots_with_errors[0].real,decimals=5)))#+" (5dp)")
plt.scatter(positive_real_roots_numerator[0].real,[0],label=r"root of expansion of $v(\alpha)$ numerator at $\alpha = $"+str(np.around(positive_real_roots_numerator[0].real,decimals=5)))#+" (5dp)")
plt.scatter(positive_real_roots_bestcase[0].real,[0],label=r"root of expansion of $v(\alpha)$ numerator w/ best-case error at $\alpha = $"+str(np.around(positive_real_roots_bestcase[0].real,decimals=5)))#+" (5dp)")
plt.xlim([.5,.69])
plt.ylim([-.75,.75])
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$v(\alpha)$")
plt.legend()
plt.scatter(test_alpha_1,0,c='k',marker='x',s=100)

plt.show()
