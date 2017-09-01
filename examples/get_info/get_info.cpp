#include "common.h"

#include <CL/cl.h>
#include <iostream>

using namespace std;

int main(void)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_device_id device = 0;
    cl_int errorNumber;    
    cl_uint unsignedInteger = 0;
    cl_bool boolean = 0;
    char charData[1024];
    cl_ulong unsignedLong = 0;
    cl_device_type deviceType = 0;

    /* Get context and while creating command queue, create device handle */
    if (!createContext(&context))
    {
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    
    /* CL_DEVICE_ADDRESS_BITS */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &unsignedInteger, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_ADDRESS_BITS " << unsignedInteger << endl;
    
    /* CL_DEVICE_AVAILABLE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &boolean, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_AVAILABLE " << boolean << endl;
    
    /* CL_DEVICE_COMPILER_AVAILABLE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &boolean, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_COMPILER_AVAILABLE " << boolean << endl;

    /* CL_DEVICE_COMPILER_AVAILABLE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &boolean, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_COMPILER_AVAILABLE " << boolean << endl;  
 
    /* CL_DEVICE_ENDIAN_LITTLE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &boolean, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_ENDIAN_LITTLE " << boolean << endl;     
    
    /* CL_DEVICE_ERROR_CORRECTION_SUPPORT */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(cl_bool), &boolean, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_ERROR_CORRECTION_SUPPORT " << boolean << endl;      
        
    /* CL_DEVICE_EXTENSIONS */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(char)*1024, &charData, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_EXTENSIONS " << charData << endl;
        
    /* CL_DEVICE_GLOBAL_MEM_CACHE_SIZE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &unsignedLong, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE " << unsignedLong << endl;

    /* CL_DEVICE_LOCAL_MEM_SIZE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &unsignedLong, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_LOCAL_MEM_SIZE " << unsignedLong << endl;
            
    /* CL_DEVICE_MAX_MEM_ALLOC_SIZE */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &unsignedLong, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE " << unsignedLong << endl;
    
    /* CL_DEVICE_VENDOR */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(char)*1024, &charData, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_VENDOR " << charData << endl;
    
    /* CL_DEVICE_VENDOR_ID */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &unsignedInteger, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_VENDOR_ID " << unsignedInteger << endl;
    
    /* CL_DEVICE_VERSION */
    if ((errorNumber = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(char)*1024, &charData, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DEVICE_VERSION " << charData << endl;
    
    /* CL_DRIVER_VERSION */
    if ((errorNumber = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(char)*1024, &charData, NULL)) != 0)
    {
        cerr << "Failed to collect device info. " << __FILE__ << ":" << __LINE__ << "->" << errorNumber << endl;
        return 1;
    }
    
    cout << "CL_DRIVER_VERSION " << charData << endl;
}
