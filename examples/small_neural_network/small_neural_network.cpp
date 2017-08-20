#include "common.h"
#include "image.h"

#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include <cstring>

using namespace std;
using namespace chrono;

#define NUM_OP 8

int main(void)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernels[NUM_OP] = {0};
    cl_event event = 0;
    
    int numberOfMemoryObjects = 10;
    cl_mem memoryObjects[10] = {0};
    cl_int errorNumber;
    string kernel_names[] = {"matrix_multiply", "matrix_nonlin", "matrix_nonlin_derivative", 
        "matrix_subtract", "matrix_point_multiply", "matrix_transpose", "matrix_multiply", "matrix_add"};
    
    size_t M = 4, N = 3, K = 1;    
    size_t globalWorksize[2] = {M, K};
    const size_t localWorksize[2] = {1, 1};
    bool setKernelArgumentsSuccess = true;
    
    /*  Prepare context, command queue, program and kernels
        NOTE: Take note, when we call cleanUpOpenCL we only send first kernel */
        
    if (!createContext(&context))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createProgram(context, device, "assets/kernels.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    for (int i = 0; i < NUM_OP; i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &errorNumber);
        
        if (!checkSuccess(errorNumber))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }
    }
    
    /* Ask the OpenCL implementation to allocate buffers for the data */     
    bool createMemoryObjectsSuccess = true;
    size_t buffSizes[] = {12, 4, 3, 4, 4, 4, 4, 4, 12, 3};
    
    for (int i = 0; i < numberOfMemoryObjects; i++)
    {    
        memoryObjects[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffSizes[i], NULL, &errorNumber);
        createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    }   
    
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL buffer. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Map the memory buffers created by the OpenCL implementation to pointers so we can access them on the CPU */
    bool mapMemoryObjectsSuccess = true;
    
    cl_float* X = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[0], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);

    cl_float* Y = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[1], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);

    cl_float* Syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], 
        CL_TRUE, CL_MAP_WRITE, 0, buffSizes[2], 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
            
    /* Initialize the data */
    cl_float _X[] = {0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
    cl_float _Y[] = {0, 0, 1, 1};
    cl_float _Syn[] = {1, -0.73692, -0.082699};
 
    for (int i=0; i<12; i++)
        X[i] = _X[i];

    for (int i=0; i<4; i++)
        Y[i] = _Y[i];
        
    for (int i=0; i<3; i++)
        Syn[i] = _Syn[i];
 
    /* Unmap buffers, so GPU can use them */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], X, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], Y, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], Syn, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    /* multiply */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 2, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 3, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 4, sizeof(cl_mem), (void*)&memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[0], 5, sizeof(cl_mem), (void*)&memoryObjects[3]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[0], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* nonlin */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void*)&memoryObjects[3]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_mem), (void*)&memoryObjects[4]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    globalWorksize[1] = K;
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* subtract */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 2, sizeof(cl_mem), (void*)&memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 3, sizeof(cl_mem), (void*)&memoryObjects[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[3], 4, sizeof(cl_mem), (void*)&memoryObjects[5]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[3], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[3], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[3], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* nonlinear derivative */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 2, sizeof(cl_mem), (void*)&memoryObjects[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[2], 3, sizeof(cl_mem), (void*)&memoryObjects[6]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[2], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* point wise */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 2, sizeof(cl_mem), (void*)&memoryObjects[5]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 3, sizeof(cl_mem), (void*)&memoryObjects[6]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[4], 4, sizeof(cl_mem), (void*)&memoryObjects[7]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[4], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* transpose */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 1, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 2, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[5], 3, sizeof(cl_mem), (void*)&memoryObjects[8]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[5], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    globalWorksize[1] = N;
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[5], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[5], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* multiply */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 0, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 2, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), (void*)&memoryObjects[8]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 4, sizeof(cl_mem), (void*)&memoryObjects[7]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[6], 5, sizeof(cl_mem), (void*)&memoryObjects[9]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    globalWorksize[0] = N;
    globalWorksize[1] = M;
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* add */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 0, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 1, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 2, sizeof(cl_mem), (void*)&memoryObjects[9]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 3, sizeof(cl_mem), (void*)&memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[7], 4, sizeof(cl_mem), (void*)&memoryObjects[2]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    globalWorksize[0] = N;
    globalWorksize[1] = K;
    
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }    
    
    /* All again */
    for (int i = 0; i<6000; i++)
    {
        globalWorksize[0] = M;
        globalWorksize[1] = K;
        
        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[0], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }
        
        globalWorksize[1] = K;
        
        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[1], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[1], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }
        
        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[3], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[3], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }

        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[2], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[2], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }

        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[4], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[4], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }

        globalWorksize[1] = N;
        
        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[5], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[5], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }

        globalWorksize[0] = N;
        globalWorksize[1] = M;
        
        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[6], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[6], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        }
        
        
        globalWorksize[0] = N;
        globalWorksize[1] = K;

        if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernels[7], 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
        {
            cleanUpOpenCL(context, commandQueue, program, kernels[7], memoryObjects, numberOfMemoryObjects);
            cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
            return 1;
        } 
    }

    /* Wait for command queue to finish, wait for event and release it */
    if (!checkSuccess(clFinish(commandQueue)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    clWaitForEvents(1, &event);  
    
    if (!checkSuccess(clReleaseEvent(event)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    /* Map buffer to read results */
    X = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[0], 0, NULL, NULL, &errorNumber);
        
    Y = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[1], 0, NULL, NULL, &errorNumber);     
        
    Syn = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[2], 0, NULL, NULL, &errorNumber);

    cl_float* L1 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[3], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[3], 0, NULL, NULL, &errorNumber);
       
    cl_float* L2 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[4], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[4], 0, NULL, NULL, &errorNumber);
        
    cl_float* L3 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[5], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[5], 0, NULL, NULL, &errorNumber);
        
    cl_float* L4 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[6], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[6], 0, NULL, NULL, &errorNumber);

    cl_float* L5 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[7], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[7], 0, NULL, NULL, &errorNumber);

    cl_float* L6 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[8], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[8], 0, NULL, NULL, &errorNumber);

    cl_float* L7 = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[9], 
        CL_TRUE, CL_MAP_READ, 0, buffSizes[9], 0, NULL, NULL, &errorNumber);     
    
    if (!checkSuccess(errorNumber))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    cout << "X: " << endl;
    for (unsigned int i = 0; i < buffSizes[0]; i++)
    {
        cout << "i= " << i << " " << X[i] << endl;
    }


    cout << "y: " << endl;
    for (unsigned int i = 0; i < buffSizes[1]; i++)
    {
        cout << "i= " << i << " " << Y[i] << endl;
    }
    
    cout << "Syn: " << endl;
    for (unsigned int i = 0; i < buffSizes[2]; i++)
    {
        cout << "i= " << i << " " << Syn[i] << endl;
    }

    cout << "L1: " << endl;
    for (unsigned int i = 0; i < buffSizes[3]; i++)
    {
        cout << "i= " << i << " " << L1[i] << endl;
    }
    
    cout << "L2: " << endl;
    for (unsigned int i = 0; i < buffSizes[4]; i++)
    {
        cout << "i= " << i << " " << L2[i] << endl;
    }

    cout << "L3: " << endl;
    for (unsigned int i = 0; i < buffSizes[5]; i++)
    {
        cout << "i= " << i << " " << L3[i] << endl;
    }

    cout << "L4: " << endl;
    for (unsigned int i = 0; i < buffSizes[6]; i++)
    {
        cout << "i= " << i << " " << L4[i] << endl;
    }

    cout << "L5: " << endl;
    for (unsigned int i = 0; i < buffSizes[7]; i++)
    {
        cout << "i= " << i << " " << L5[i] << endl;
    }
    
    cout << "L6: " << endl;
    for (unsigned int i = 0; i < buffSizes[8]; i++)
    {
        cout << "i= " << i << " " << L6[i] << endl;
    }

    cout << "L7: " << endl;
    for (unsigned int i = 0; i < buffSizes[9]; i++)
    {
        cout << "i= " << i << " " << L7[i] << endl;
    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[3], L1, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    cleanUpOpenCL(context, commandQueue, program, kernels[0], memoryObjects, numberOfMemoryObjects);
}
