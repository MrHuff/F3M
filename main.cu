/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
//#include <stdio.h>
#include "matrix.cu"

/*
*********************************************************************
function name: main
description: test and compare
parameters:
            none
return: none
*********************************************************************
*/

int main(int argc, char const *argv[])
{
    int n = 10;
    int m = 10;
    cumatrix<float> a(n, m), b(n, m),f(1,1); //this calls the constructors
    cumatrix<float> c = a * b; //Do matmul, constructor, we initialize a new cumatrix<float> which is c in the function.
//    f = c; //this calls the copy ass
//    cumatrix<float> bub(a); //this calls the copy constructor
//    cumatrix<float> foo = test_move_constructor(cumatrix<float>(1,1)); //this calls the move constructor
//    c =  test_move_constructor(cumatrix<float>(1,1)); //this calls the move assignment op
    return 0;
}