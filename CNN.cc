#include "CNN.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

//creates graph of operations for an image, does not execute them
//bool value should be false if done for one image, true if for loading a batch
Status CNN::CreateGraphForImage(bool unstack)
{
    //placeholder tensor to be replaced by the feed mechanism
    //first arg: scope - we create a child scope from root 
    //second arg: tensor data type, i.e. it will be a tensor with string elements
    file_name_var = Placeholder(image_root.WithOpName("input"), DT_STRING);
    //operation which reads file - a new node in TensorFlow graph which is connected to previous operation
    //by an edge - tensor file_name_var
    auto file_reader = ReadFile(image_root.WithOpName("file_readr"), file_name_var);


    //decoding a JPEG-encoded image to a uint8 tensor
    //third arg: setting the number of color channels in the decoded image to 3, which outputs an RGB image
    auto image_reader = DecodeJpeg(image_root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(image_channels));


    //cast image data to float for math operations (typical for TensorFlow)
    auto float_caster = Cast(image_root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    //convention in TensorFlow - images are expected to be send in batches, so they are four dimensional arrays with indices of 
    //[batch index Number, Height of the image, Width of the image, image Channels] aka NHWC (functions expect it etc.)
    //so we have to add a batch dimension to the start with ExpandDims
    //third arg: index in shape array of resulting tensor (0 means the added dimension is at the start) 
    auto dims_expander = ExpandDims(image_root.WithOpName("dim"), float_caster, 0);
    //resize image to size specified in input parameters with bilinear interpolation - this is done so that
    //all images fed to the network will have the same size
    //third arg: tensor-vector wth two elements: height and width - Const operation creates such a tensor
    auto resized = ResizeBilinear(image_root.WithOpName("size"), dims_expander, Const(image_root, {image_height, image_width}));
    //normalization of float elements to values between 0 and 1; performs two mathematical operations in one:
    //subtracts the mean (argument image_mean) and divides by the scale (argument image_std)
    //div = (resized - image_mean) / image_std
    auto div = Div(image_root.WithOpName("normalized"), Sub(image_root, resized, {image_mean}), {image_std});


    //check if tensors should be unstacked
    if(unstack)
    {
        //Unstack unpacks one R+1-rank tensor into N R-rank tensors
        //while stacking an additional dimension is added; so as to not overdo it, a batch dimension is unpacked 
        auto output_list = Unstack(image_root.WithOpName("fold"), div, 1);
        //image_tensor_var contains a tensor with specified image/first image in batch 
        image_tensor_var = output_list.output[0];
    }
    else
        image_tensor_var = div;
    return image_root.status();
}

//runs the graph created by CreateGraphForImage function; in this way creates image tensor from file
Status CNN::ReadTensorFromImageFile(const string& file_name, Tensor& out_tensor)
{
    //decoding a JPEG-encoded image to a uint8 tensor; if file doesn't have a jpg/jpeg extension throw error
    //third arg: setting the number of color channels in the decoded image to 3, which outputs an RGB image
    if (!str_util::EndsWith(file_name, ".jpg") && !str_util::EndsWith(file_name, ".jpeg"))
        return errors::InvalidArgument("Only jpg files are accepted");

    //create a client-session, run it and send results; create a vector of tensors for results 
    ClientSession session(image_root);
    vector<Tensor> out_tensors;
    //first arg: list of pairs with index and element; one pair is passed, with variable
    //node from Placeholder file_name_var as index and a value of string file_name as element
    //second arg: vector of graph nodes to evaluate
    //third arg: vector of tensors, results of evaluation
    //TF_CHECK_OK macro for errors etc.
    TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {image_tensor_var}, &out_tensors));
    //make a shallow copy for output 
    out_tensor = out_tensors[0];
    return Status::OK();
}

/*reads all files from specified subdirectories and returns a shuffled vector of pairs with image tensors created through
  ReadTensorFromImageFile function and a specified label; assumes graph for said action is already created   

  it assumes a folder structure where in a base folder there are subfolders with image files placed in them according to their label and nothing
  else in them; in our example it looks like this: 

.../train/
        cats/
            cat.0.jpg
            cat.1.jpg
            cat.2.jpg
            ...
            cat.999.jpg
        dogs/
            dog.0.jpg
            dog.1.jpg
            dog.2.jpg
            ...
            dog.999.jpg

  first arg: path to base folder (train in example above)
  second arg: vector of pairs of [subfolder name, label value]; labels should be different floats
              one way of passing it to function is: {make_pair("cats", 0), make_pair("dogs", 1)}
  third arg: returning vector of pairs, each one is a tensor of an image and a label                
*/
Status CNN::ReadFileTensors(string& base_folder_path, vector<pair<string, float>> subfolder_label_pairs, vector<pair<Tensor, float>>& file_tensors)
{
    //utility class which gives similar facilitation like C++17 std::filesystem to operate on files etc.
    Env* penv = Env::Default();
    //validate the folder; control if path leads to a directory on all relevant levels going forward 
    TF_RETURN_IF_ERROR(penv->IsDirectory(base_folder_path));

    //flag for shuffling returning vector of pairs; becomes true when first category is fully loaded
    bool shuffle = false;

    //for each image category
    for (pair<string, float> pair : subfolder_label_pairs)
    {

        //JoinPath - API function for concatenating path strings, subfolder_path <- full path to subfolder
        string subfolder_path = io::JoinPath(base_folder_path, pair.first);
        TF_RETURN_IF_ERROR(penv->IsDirectory(subfolder_path));

        //store names of all files in the subfolder in a vector
        vector<string> file_names;
        TF_RETURN_IF_ERROR(penv->GetChildren(subfolder_path, &file_names));

        //for each image file
        for(string file: file_names)
        {
            //full_path <- full path to image call
            string full_path = io::JoinPath(subfolder_path, file);
            //call ReadTensorFromImageFile function on said image file, assumes graph is already created
            Tensor image_tensor;
            TF_RETURN_IF_ERROR(ReadTensorFromImageFile(full_path, image_tensor));

            size_t s = file_tensors.size();
            //shuffling is used to feed differing images in succession into the net (i.e. avoid situation where first all cats are fed and then all dogs)
            if(shuffle)
            {
                //shuffle the images
                int i = rand() % s;
                file_tensors.emplace(file_tensors.begin()+i, make_pair(image_tensor, pair.second));
            }
            else
                file_tensors.push_back(make_pair(image_tensor, pair.second));
        }
        //becomes true when first category is fully loaded - no need for shuffling beforehand
        shuffle = true;
    }
    return Status::OK();
}







/*
//function which reproduces image from a tensor - reverse operation to ReadTensorFromImageFile
Status WriteTensorToImageFile(const string& file_name, const int input_height,
                              const int input_width, const float input_mean,
                              const float input_std, vector<Tensor>& in_tensors)
{
    auto root = Scope::NewRootScope();
    //reversing normalisation operation on input tensor
    //resized = d * input_std + input_mean <=> un_normalized = in_tensors[0] * input_std + input_mean
    //tutorial has an error here - wrong order of operations, which does not matter as long as input_mean=0
    //(which it is set to in main)
    //auto un_normalized = Multiply(root.WithOpName("un_normalized"), Add(root, in_tensors[0], {input_mean}), {input_std});
    auto un_normalized = Add(root.WithOpName("un_normalized"), Multiply(root, in_tensors[0], {input_std}), {input_mean}); 
    //reshape to original size - 3-dimensional tensor with height and width passed as arguments and RGB values in 3rd dim    
    auto shaped = Reshape(root.WithOpName("reshape"), un_normalized, Const(root, {input_height, input_width, 3}));
    if(!root.ok())
        LOG(FATAL) << root.status().ToString();
    //recast to uint8
    auto casted = Cast(root.WithOpName("cast"), shaped, DT_UINT8);
    //encode as JPEG
    auto image = EncodeJpeg(root.WithOpName("EncodeJpeg"), casted);

    //create ClientSession and run
    vector<Tensor> out_tensors;
    ClientSession session(root);
    TF_CHECK_OK(session.Run({image}, &out_tensors));

    //write image into a file using ofstream
    ofstream fs(file_name, ios::binary);
    //tstring instead of string, types were changed in TensorFlow
    fs << out_tensors[0].scalar<tstring>()();
    return Status::OK();
}*/


int main(int argc, const char * argv[])
{
    /*
    //set image data
    string image = "/home/mordoksieznik/Code/tensorflow/tensorflow/examples/my_project1/data/test.jpg";
    int32 input_width = 299;
    int32 input_height = 299;
    float input_mean = 10;
    float input_std = 255;
    vector<Tensor> resized_tensors;

    //read tensor from image, reverse this operation and check results 
    Status read_tensor_status = ReadTensorFromImageFile(image, input_height, input_width, input_mean,
                                input_std, &resized_tensors, true);
    cout << resized_tensors[0].shape().DebugString() << endl;
    if (!read_tensor_status.ok())
    {
        LOG(ERROR) << read_tensor_status;
        return -1;
    }
    Status write_tensor_staus = WriteTensorToImageFile(
                                "/home/mordoksieznik/Code/tensorflow/tensorflow/examples/my_project1/data/output.jpg",
                                input_height, input_width, input_mean, input_std, resized_tensors);
    */
    return 0;
}