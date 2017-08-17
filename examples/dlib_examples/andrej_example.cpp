#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <fstream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------
float func(string f_path, string s_path)
{

    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> f_img, s_img;
    load_image(f_img, f_path);
    load_image(s_img, s_path);


    // Display the raw image on the screen
    //image_window f_win(f_img); 
    //image_window s_win(s_img); 

    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(f_img))
    {
        auto shape = sp(f_img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(f_img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
    }
    
    for (auto face : detector(s_img))
    {
        auto shape = sp(s_img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(s_img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
    }

    if (faces.size() < 2)
    {
        cout << "No faces .-!" << endl;
        
        // Here we try to find face again on same image, just double in size
        pyramid_up(f_img);
        pyramid_up(s_img);

        for (auto face : detector(f_img))
        {
            auto shape = sp(f_img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(f_img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
        }
        
        for (auto face : detector(s_img))
        {
            auto shape = sp(s_img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(s_img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
        }

        if (faces.size() < 2)
        {
            cout << "No faces found!" << endl;
            return -1;
        }

    }
    else if (faces.size() > 2)
    {
        cout << "More than two faces find!" << endl;
    }
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);
    
    std::vector<float> results;
    float min = 10.0;
    
    for (int i=0; i<faces.size()-1; i++)
    {    
        for (int j=i+1; j<faces.size(); j++)
        {
            if (length(face_descriptors[i]-face_descriptors[j]) < min)
                min = length(face_descriptors[i]-face_descriptors[j]);
        }
    }
    
    
    return min;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently.
    thread_local random_cropper cropper;
    cropper.set_chip_dims(150,150);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_height(0.99999);
    cropper.set_background_crops_fraction(0);
    cropper.set_min_object_height(0.97);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);
    std::vector<matrix<rgb_pixel>> crops; 

    matrix<rgb_pixel> temp; 
    for (int i = 0; i < 100; ++i)
    {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}

string prepare(string name, string number) 
{
    const string path = "/home/linaro/Desktop/lfw/";
    string image;    

    if (number.length() < 2)
        image = "_000" + number;
    else if (number.length() < 3)
        image = "_00" + number;
    else 
        image = "_0" + number;

    //cout << path + name + "/" + name + image + ".jpg" << flush;
    return path + name + "/" + name + image + ".jpg";   
}

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv) 
try
{
    // Open file containing references to testing pairs and read line by line
    ifstream infile("/home/linaro/Desktop/pairs.txt");
    string f_name, s_name, f_num, s_num;

    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }
    
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
    //-------------------------------------------------------------------------------------
    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(f_name, s_num)) << endl;
    }

    for (int i=0; i<300; i++)
    {
        infile >> f_name >> f_num >> s_name >> s_num;
        cout << i << " " << func(prepare(f_name, f_num), prepare(s_name, s_num)) << endl;
    }
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}
