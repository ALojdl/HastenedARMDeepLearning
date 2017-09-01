#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

/*
int main()
{
    matrix<double,32,32> input;
    matrix<double,28,28> output;
    matrix<double,5,5> filter;   
    
    double acc;
    
    input = 1;
    filter = 0.1;
    
    for (int i=0; i<28; i++)
    {
        for (int j=0; j<28; j++)
        {
            acc = 0;
            for (int r=0; r<5;  r++)
            {
                for (int k=0; k<5; k++)
                {
                    acc += input(i+r,j+k) * filter(r,k);
                }
            }
            output(i,j) = acc/25;
        }
    }
    cout << input << endl;
    cout << output << endl; 
}
*/


int main()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    
    matrix<double,1,84> input;
    matrix<double,84,10> syn, tmp_syn, tmp;
    matrix<double,1,10> output, y, a, e, g, delta;
    
    for (int i=0; i<84; i++)
        for (int j=0; j<10; j++)
            syn(i,j) = distribution(generator);
    
    output(0) = 0;
    output(1) = 1;
    output(2) = 0;
    output(3) = -1;
    output(4) = 0;
    output(5) = 1;
    output(6) = 0;
    output(7) = -1;
    output(8) = 0;
    output(9) = 1;
    
    input = 1;
    
    // Learning 
    for (int i=0; i<100000; i++)
    {
        y = input * syn;
        a = 1/(1+exp(-y));
        e = output - a;
        g = pointwise_multiply(a,(1 - a));
        delta = pointwise_multiply(g, e);
        //tmp = syn;
        //tmp_syn = trans(input) * delta;
        //syn = tmp + tmp_syn;
        syn += trans(input) * delta;
    }
    cout << syn << endl;   
}


