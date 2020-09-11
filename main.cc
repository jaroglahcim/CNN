#include <iostream>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/summary/summary_file_writer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

//function which "reads an image, does some manipulation to it and returns a tensor you can feed another graph to
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors,
                               bool writeGraph)
{
    //create new scope
    auto root = Scope::NewRootScope();
    //placeholder tensor to be replaced by the feed mechanism
    //first arg: scope - we create a child scope from root 
    //second arg: tensor data type, i.e. it will be a tensor with string elements
    auto file_name_var = Placeholder(root.WithOpName("input"), DT_STRING);
    //operation which reads file - a new node in TensorFlow graph which is connected to previous operation
    //by an edge - tensor file_name_var
    auto file_reader = ReadFile(root.WithOpName("file_readr"), file_name_var);


    //decoding a JPEG-encoded image to a uint8 tensor; if file doesn't have a jpg extension throw error
    //third arg: setting the number of color channels in the decoded image to 3, which outputs an RGB image
    if (!str_util::EndsWith(file_name, ".jpg"))
        return errors::InvalidArgument("Only jpg files are accepted");
    const int wanted_channels = 3;
    auto image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(wanted_channels));


    //cast image data to float for math operations (typical for TensorFlow)
    auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    //convention in TensorFlow - images are expected to be send in batches, so they are four dimensional arrays with indices of 
    //[batch index Number, Height of the image, Width of the image, image Channels] aka NHWC (functions expect it etc.)
    //so we have to add a batch dimension of 1 to the start with ExpandDims
    //third arg: index in shape array of resulting tensor (0 means the added dimension is at the start) 
    auto dims_expander = ExpandDims(root.WithOpName("dim"), float_caster, 0);
    //resize image to size specified in input parameters with bilinear interpolation - this is done so that
    //all images fed to the network will have the same size
    //third arg: tensor-vector wth two elements: height and width - Const operation creates such a tensor
    auto resized = ResizeBilinear(root.WithOpName("size"), dims_expander, Const(root, {input_height, input_width}));
    //normalization of float elements to values between 0 and 1; performs two mathematical operations in one:
    //subtracts the mean (argument input_mean) and divides by the scale (argument input_std)
    //d = (resized - input_mean) / input_std
    auto d = Div(root.WithOpName("normalized"), Sub(root, resized, {input_mean}), {input_std});


    //Now that graph was created, we create a client-session...
    ClientSession session(root);
    //and run it and get results; first arg: list of pairs with index and element; one pair is passed, with variable
    //node from Placeholder file_name_var as index and a value of string file_name as element
    //second arg: vector of graph nodes to evaluate - in this case result of operations defined above, d
    //third arg: vector of tensors, results of evaluation - in this case only one tensor is expected
    //TF_CHECK_OK macro ????? 
    TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {d}, out_tensors));


    //creating graph visualisation using TensorBoard, if bool option in function arguments set to true 
    if(writeGraph)
    {
        //defining graph variable
        GraphDef graph;
        //extract graph object from scope using utility function ToGraphDef
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
        //creating SummaryFileWriter, which will write graph files
        //first arg: max queue of graphs before writing, in this case 1
        //second arg: wait time in milliseconds, in this case 0
        //third arg: path to folder where files will be stored
        //fourth arg: file name suffix
        //sixth arg: resulting file writer
        SummaryWriterInterface* w;
        TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/home/mordoksieznik/Code/tensorflow/tensorflow/examples/my_project1/graphs",
                    ".img-graph", Env::Default(), &w));
        //using created writer to write said graph
        TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
    }
    return Status::OK();
}


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
}


int main(int argc, const char * argv[])
{
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
    return 0;
}