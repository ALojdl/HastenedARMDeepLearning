#include "common.h"
#include "image.h"

#include <CL/cl.h>
#include <iostream>
#include <chrono>

#define SIZE 128
#define LOCAL_SIZE 16
#define WORK_SIZE(X, Y) ((X) < (Y) ? (X) : (Y/2))
#define WPT 1

using namespace std;
using namespace chrono;

int main(void)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_event event = 0;
    
    int numberOfMemoryObjects = 3;
    cl_mem memoryObjects[3] = {0, 0, 0};
    cl_int errorNumber;
    
    size_t M, N, K;
    M = N = K = SIZE;
    
    size_t globalWorksize[2] = {M, N/WPT};
    const size_t localWorksize[2] = {WORK_SIZE(LOCAL_SIZE, SIZE), WORK_SIZE(LOCAL_SIZE, SIZE)/WPT};
    cout << WORK_SIZE(LOCAL_SIZE, SIZE);
    cl_int arraySize = SIZE * SIZE;
    size_t bufferSize = arraySize * sizeof(cl_float);
    bool setKernelArgumentsSuccess = true;
    
    steady_clock::time_point begin, exec, end;

    /* Remmember start time */
    begin = steady_clock::now();
    
    /* Prepare context, command queue, program and kernels */
    if (!createContext(&context))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createProgram(context, device, "assets/multiply.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    kernel = clCreateKernel(program, "matrix_multiply_less_loads", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Ask the OpenCL implementation to allocate buffers for the data */     
    bool createMemoryObjectsSuccess = true;
    
    memoryObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    memoryObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    memoryObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL buffer. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Map the memory buffers created by the OpenCL implementation to pointers so we can access them on the CPU */
    bool mapMemoryObjectsSuccess = true;
    
    cl_float* A = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], 
        CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);

    cl_float* B = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], 
        CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    
    /* Initialize the data */
    for (int i = 0; i < arraySize; i++)
    {
       A[i] = 1.0f;
       B[i] = 1.0f;
    }
    
    /* Unmap buffers, so GPU can use them */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], A, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], B, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    /* Set the kernel arguments for first matrix multiply and enqueue the kernel */
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(int), (void*)&M));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(int), (void*)&N));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(int), (void*)&K));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&memoryObjects[2]));
   
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* Kernel is pushed to queue, remmember time */
    exec = steady_clock::now();

    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorksize, localWorksize, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for command queue to finish, wait for event and release it */
    if (!checkSuccess(clFinish(commandQueue)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    clWaitForEvents(1, &event);  
    
    end = steady_clock::now();  
    
    if (!checkSuccess(clReleaseEvent(event)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    /* Map buffer to read results */
    cl_float* C = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], 
        CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, NULL, NULL, &errorNumber);
    
    if (!checkSuccess(errorNumber))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }

    cout << "C: " << endl;
    for (int i = 0; i < arraySize; i++)
    {
/*
        if (C[i] != K)
            cout << "Error, result is " << C[i] << " not a " << K << " at index " << i << endl;
*/
        cout << "i= " << i << " " << C[i] << endl;

    }
    
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], C, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    
    /* Print timming information */
    cout << "Prepare time " << duration_cast<chrono::microseconds> (exec - begin).count() << " us" << endl;
    cout << "Execution time " << duration_cast<chrono::microseconds> (end - exec).count() << " us" << endl;

    cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
}
