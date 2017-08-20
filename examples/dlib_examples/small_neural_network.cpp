#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <dlib/matrix.h>


using namespace dlib;
using namespace std;

double nonlin(double x, bool deriv)
{
    if (deriv)
        return x * (1 - x);
    else
        return 1/(1+exp(-x));
}
/*
template<class T> T nonlinT(T x, bool deriv)
{
    if (deriv)
        return x * (1 - x);
    else
        return 1/(1+exp(-x));
}


void foo(std::vector<int> x);

foo({1,2,3,4});
*/

int main()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);

    matrix<double,4,3> X, l0;
    matrix<double,0,1> y, l1, l1_error, l1_delta, l1_tmp;
    matrix<double,3,1> syn;
    
    y.set_size(4);
    l1.set_size(4);
    l1_error.set_size(4);
    l1_delta.set_size(4);
    l1_tmp.set_size(4);
    
    X = 0, 0, 1,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1;

    y = 0,
        0, 
        1,
        1;
        
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    for (int i=0; i<3; i++)
        syn(i,1) = distribution(generator);
        
    cout << syn;
   

    for (int i=0; i<60000; i++)
    {
        l0 = X;
        l1 = l0 * syn;
    
        for (int i=0; i<4; i++)
            l1(i) = nonlin(l1(i), false);   

        l1_error = y - l1;

        for (int i=0; i<4; i++)
            l1_tmp(i) = nonlin(l1(i), true);   

        l1_delta = pointwise_multiply(l1_error, l1_tmp);

        syn += trans(l0) * l1_delta;
    }
    
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Exec: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << endl;
    
    cout << "syn: \n" << syn << endl;
    cout << "l1: \n" << l1 << endl;

}

