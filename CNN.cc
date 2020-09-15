#include "CNN.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace chrono;

//                                                     LOADING IMAGES FUNCTIONS

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

    auto div = Div(image_root.WithOpName("normalized"), resized, {255.f});
    //auto div = Div(image_root.WithOpName("normalized"), Sub(image_root, resized, {image_mean}), {image_std});


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
    //(Placeholder is like a function parameter, it is specified here with its value)   
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
    for (auto pair : subfolder_label_pairs)
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

//reads files from base folder in labeled subfolders and creates vectors of tensors of batches of images and labels
//check ReadFileTensors function comments for further info on folder structure
//first arg: path to base folder
//second arg: vector of pairs of [subfolder name, label value]; labels should be different floats
//third arg: batch size, i.e. how many images should be fed into the net in one go
//fourth arg: returning vector of image batches tensors
//fifth arg: returning vector of label batches tensors
//4th and 5th arg have same index number
Status CNN::ReadBatches(string& base_folder_path, vector<pair<string, float>> subfolder_label_pairs, int batch_size,
                        vector<Tensor>& image_batches, vector<Tensor>& label_batches)
{
    //reads the folder and its sub-folders’ content into a vector pair using ReadFileTensors
    vector<pair<Tensor, float>> all_files_tensors;
    TF_RETURN_IF_ERROR(ReadFileTensors(base_folder_path, subfolder_label_pairs, all_files_tensors));

    //variables for splitting batches 
    auto start_image = all_files_tensors.begin();
    auto end_image = all_files_tensors.begin()+batch_size;

    size_t batches = all_files_tensors.size()/batch_size;
    if(batches*batch_size < all_files_tensors.size())
        batches++;

    //for each batch
    for(int b = 0; b < batches; b++)
    {
        //checking if end_image did not go overboard
        if(end_image > all_files_tensors.end())
            end_image = all_files_tensors.end();

        //extracts batch from the whole vector of pairs 
        vector<pair<Tensor, float>> one_batch(start_image, end_image);
        //need to break the pairs to create two Input vectors where each n-th element in the Tensor vector matches the n-th element in the labels vector
        vector<Input> one_batch_image, one_batch_label;
        //for each pair in batch
        for(auto pair: one_batch)
        {
            //add to image vector
            one_batch_image.push_back(Input(pair.first));
            //add to label vector - need to convert it first through a Tensor
            Tensor t(DT_FLOAT, TensorShape({1}));
            t.scalar<float>()(0) = pair.second;
            one_batch_label.push_back(Input(t));
        }

        //creating a tiny graph and running it (for each batch as we are in a loop)
        //the operation that will be performed - Stack, which accepts an InputList as an argument
        InputList one_batch_inputs(one_batch_image);
        InputList one_batch_labels(one_batch_label);
        //stacking said InputLists and running a session, storing a result in a vector of Tensors out_tensors
        Scope root = Scope::NewRootScope();
        //Stack creates a single 4d tensor out of our 3d image vectors
        auto stacked_images = Stack(root, one_batch_inputs);
        auto stacked_labels = Stack(root, one_batch_labels);
        TF_CHECK_OK(root.status());
        ClientSession session(root);
        vector<Tensor> out_tensors;
        TF_CHECK_OK(session.Run({}, {stacked_images, stacked_labels}, &out_tensors));

        //adding created tensors to returning vectors of tensors
        image_batches.push_back(out_tensors[0]);
        label_batches.push_back(out_tensors[1]);
        start_image = end_image;
        if(start_image == all_files_tensors.end())
            break;
        end_image = start_image + batch_size;
    }
    return Status::OK();
}


//                                                          NETWORK FUNCTIONS


//function which creates a single convolution layer graph with Rectified Linear Unit (ReLU) activation function with 1 stride and same-type padding plus
//max pooling using a 2x2 window and stride 2, initialized with Xavier initialization function and 0 as bias
//first arg: index string to distinguish between layers
//second arg: subscope - since layers are a composition of operations in low lever API, a subscope is created for each layer
//?
Input CNN::AddConvolutionLayer(string index, Scope scope, int in_channels, int out_channels, int filter_height, int filter_width, Input input)
{
    //create a TensorShape denoted by its number of dimensions and size for each dimension
    //in this example 4d shape with sizes denoted by variables: filter height and width, in and out channels
    TensorShape sp({filter_height, filter_width, in_channels, out_channels});
    //Conv2D operation needs a Tensor variable which holds different filters (32 in first layer), which will be changed with each step when
    //network is in training; each filter is represented by a 4d tensor, shape of which is specified by sp 
    //map_vars map holds these variables with string key with W in front (and biases with B instead)
    map_vars["W"+index] = Variable(scope.WithOpName("W"), sp, DT_FLOAT);
    //map_shapes map holds shapes of map_vars variable
    map_shapes["W"+index] = sp;
    //map_assigns map stores operations with which variables are initialized - it is done using Xavier initialization method
    map_assigns["W"+index+"_assign"] = Assign(scope.WithOpName("W_assign"), map_vars["W"+index], XavierInitialization(scope, in_channels, out_channels, filter_height, filter_width));
    
    sp = {out_channels};
    map_vars["B"+index] = Variable(scope.WithOpName("B"), sp, DT_FLOAT);
    map_shapes["B"+index] = sp;
    //bias is initialized with a 0
    map_assigns["B"+index+"_assign"] = Assign(scope.WithOpName("B_assign"), map_vars["B"+index], Input::Initializer(0.f, sp));

    //creating a graph of operations to be done in a layer

    //convolution-filter operation with the specified input, filter variable, third arg: array of strides (we use standard 1)
    //fourth arg: padding type name (we use standard SAME)
    auto conv = Conv2D(scope.WithOpName("Conv"), input, map_vars["W"+index], {1, 1, 1, 1}, "SAME");
    //add bias after filtering operation
    auto bias = BiasAdd(scope.WithOpName("Bias"), conv, map_vars["B"+index]);
    //perform relu activation function
    auto relu = Relu(scope.WithOpName("Relu"), bias);
    //pooling max value of each 2x2 window with stride 2; stride and window should keep 1 on first and last array place and change middle ones only
    //(in python you specify only those) 
    return MaxPool(scope.WithOpName("Pool"), relu, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");
}

//function which returns initialization according to Xavier method ?
Input CNN::XavierInitialization (Scope scope, int in_channels, int out_channels, int filter_height, int filter_width)
{
    float std;
    Tensor t;
    if(filter_height == 0)
    { //Dense
        std = sqrt(6.f/(in_channels+out_channels));
        Tensor ts(DT_INT64, {2});
        auto v = ts.vec<int64>();
        v(0) = in_channels;
        v(1) = out_channels;
        t = ts;
    }
    else
    { //Conv
        std = sqrt(6.f/(filter_height*filter_width*(in_channels+out_channels)));
        Tensor ts(DT_INT64, {4});
        auto v = ts.vec<int64>();
        v(0) = filter_height;
        v(1) = filter_width;
        v(2) = in_channels;
        v(3) = out_channels;
        t = ts;
    }
    auto rand = RandomUniform(scope, t, DT_FLOAT);
    return Multiply(scope, Sub(scope, rand, 0.5f), std*2.f);
}

//function which creates a single dense layer graph; works largely analogously to AddConvolutionLayer function
//fifth arg: bool relu_activation determines if ReLU activation will be performed or should binary activation be used
//ReLU is used on all dense layers besides the last one
Input CNN::AddDenseLayer(string index, Scope scope, int in_units, int out_units, bool relu_activation, Input input)
{
    //map_vars holds weights for input multiplication and bias to add
    TensorShape sp = {in_units, out_units};
    map_vars["W"+index] = Variable(scope.WithOpName("W"), sp, DT_FLOAT);
    map_shapes["W"+index] = sp;
    //weights here are also initialized with Xavier initialization
    map_assigns["W"+index+"_assign"] = Assign(scope.WithOpName("W_assign"), map_vars["W"+index], XavierInitialization(scope, in_units, out_units));
    sp = {out_units};
    map_vars["B"+index] = Variable(scope.WithOpName("B"), sp, DT_FLOAT);
    map_shapes["B"+index] = sp;
    //biases are also initialized with 0
    map_assigns["B"+index+"_assign"] = Assign(scope.WithOpName("B_assign"), map_vars["B"+index], Input::Initializer(0.f, sp));
    //here multiplication and adding bias is performed
    auto dense = Add(scope.WithOpName("Dense_b"), MatMul(scope.WithOpName("Dense_w"), input, map_vars["W"+index]), map_vars["B"+index]);
    //in the last dense layer ReLU activation is skipped
    if(relu_activation)
        return Relu(scope.WithOpName("Relu"), dense);
    else
        return dense;
}


//function which creates full CNN architecture graph (without backpropagation
//4 Convergence Layers with Maxpooling and 3 Dense Layers (2 with ReLU activation, last with sigmoid function) with Dropout (for dense layers obv)  
Status CNN::CreateGraphForCNN(int filter_height, int filter_width){
    //input batch of images, it's size in our example is batch_sizex150x150x3
    //again starting with a placeholder - we again keep this variable stored in object, so as to avoid trouble with passing it later 
    input_batch_var = Placeholder(net_root.WithOpName("input_batch"), DT_FLOAT);
    //drop_rate and skip_drop are used for dropout technique used further down
    drop_rate_var = Placeholder(net_root.WithOpName("drop_rate"), DT_FLOAT);
    skip_drop_var = Placeholder(net_root.WithOpName("skip_drop"), DT_FLOAT);
    
    //First Layer:
    //Start Conv+Maxpool No 1. filter size 3x3x3 and we have 32 filters
    Scope scope_conv1 = net_root.NewSubScope("Conv1_layer");
    int in_channels = image_channels;
    int out_channels = 32;
    auto pool1 = AddConvolutionLayer("1", scope_conv1, in_channels, out_channels, filter_height, filter_width, input_batch_var);
    //controlling size
    int new_width = ceil((float)image_width / 2); //max pool is reducing the size by factor of 2
    int new_height = ceil((float)image_height / 2); //max pool is reducing the size by factor of 2

    //Conv+Maxpool No 2, filter size still 3x3x3; usually filter count increases as we go through layers, here there are 64 filters
    Scope scope_conv2 = net_root.NewSubScope("Conv2_layer");
    in_channels = out_channels;
    out_channels = 64;
    auto pool2 = AddConvolutionLayer("2", scope_conv2, in_channels, out_channels, filter_height, filter_width, pool1);
    new_width = ceil((float)new_width / 2);
    new_height = ceil((float)new_height / 2);

    //Conv+Maxpool No 3
    Scope scope_conv3 = net_root.NewSubScope("Conv3_layer");
    in_channels = out_channels;
    out_channels = 128;
    auto pool3 = AddConvolutionLayer("3", scope_conv3, in_channels, out_channels, filter_height, filter_width, pool2);
    new_width = ceil((float)new_width / 2);
    new_height = ceil((float)new_height / 2);


    //Conv+Maxpool No 4
    Scope scope_conv4 = net_root.NewSubScope("Conv4_layer");
    in_channels = out_channels;
    out_channels = 128;
    auto pool4 = AddConvolutionLayer("4", scope_conv4, in_channels, out_channels, filter_height, filter_width, pool3);
    new_width = ceil((float)new_width / 2);
    new_height = ceil((float)new_height / 2);

    //Flatten
    //reshaping data so that it will be stored in a 2d tensor - batch number and data, suitable for use in dense layers
    Scope flatten = net_root.NewSubScope("flat_layer");
    //calculating flat length of data for use in reshape function 
    int flat_len = new_width * new_width * out_channels;
    auto flat = Reshape(flatten, pool4, {-1, flat_len});
    
    //Dropout
    //the most popular regularization technique; in it randomly some neurons are deactivated each iteration during training so that whole
    //network operates with less dependence on a few neurons afterwards (during validation/prediction), which prevents overfitting
    //drop_rate_var and skip_drop_var have to be set beforehand to correct values to use
    //standard uses - drop_rate_var <- 0.5f, skip_drop_var <- 0.f during training,  drop_rate_var <- 1.f, skip_drop_var <- 1.f during validation/prediction
    //(has to be used in such a cumbersome way because with switch and merge backpropagation is impossible)   
    Scope dropout = net_root.NewSubScope("Dropout_layer");
    //Uniform distribution, args: scope, shape of the output tensor, type of output; output tensor is of size of flattened image batch
    //and have random floating point numbers with uniform distribution from [0, 1] range
    auto rand = RandomUniform(dropout, Shape(dropout, flat), DT_FLOAT);
    //binary = floor(rand + (1 - drop_rate) + skip_drop)
    //returns a vector of same size with binary values of 0 and 1 through flooring in such way that about a skip_drop percentage of
    //elements in tensor have a value of 1 (bar the situation of skipping drop)
    auto binary = Floor(dropout, Add(dropout, rand, Add(dropout, Sub(dropout, 1.f, drop_rate_var), skip_drop_var)));
    //when dropping, remaining neurons have values increased by dividing by the drop rate
    auto after_drop = Multiply(dropout.WithOpName("dropout"), Div(dropout, flat, drop_rate_var), binary);

    //Dense No 1
    int in_units = flat_len;
    int out_units = 512;
    Scope scope_dense1 = net_root.NewSubScope("Dense1_layer");
    auto relu5 = AddDenseLayer("5", scope_dense1, in_units, out_units, true, after_drop);

    //Dense No 2
    in_units = out_units;
    out_units = 256;
    Scope scope_dense2 = net_root.NewSubScope("Dense2_layer");
    auto relu6 = AddDenseLayer("6", scope_dense2, in_units, out_units, true, relu5);
    
    //Dense No 3
    in_units = out_units;
    out_units = 1;
    Scope scope_dense3 = net_root.NewSubScope("Dense3_layer");
    //bool relu_activation changes to false - on last layer binary activation is used
    auto logits = AddDenseLayer("7", scope_dense3, in_units, out_units, false, relu6);

    //sigmoid function for binary classification
    out_classification = Sigmoid(net_root.WithOpName("output_classes"), logits);
    return net_root.status();
}

//adds nodes to neural net graph created in CreateGraphForCNN function to "optimize it" - adds backpropagation to it
//regular mean square difference is used for loss calculation and Adam Optimization Algorithm is used for optimization
//(instead of gradient descent or whatever)
//first arg: IMPORTANT pass number for wildly differing results, one that worked well is 0.0001    
Status CNN::CreateOptimizationGraph(float learning_rate)
{
    //Placeholder for labels
    input_labels_var = Placeholder(net_root.WithOpName("inputL"), DT_FLOAT);

    //Adding to graph operation for loss calculation - mean square difference
    //cross entropy and SoftmaxCrossEntropyWithLogits function could be better, but not know how to use
    //in out_classification are our results from previous classification batch 
    Scope scope_loss = net_root.NewSubScope("Loss_scope");
    out_loss_var = Mean(scope_loss.WithOpName("Loss"), SquaredDifference(scope_loss, out_classification, input_labels_var), {0});
    TF_CHECK_OK(scope_loss.status());

    //AddSymbolicGradients takes graph from scope, adds all relevant backpropagating operations and returns a vector
    //of gradient of same size as number of weights and biases, i.e. calculates gradient
    //third arg: vector of weights and biases; it is created from the map map_vars which was set to be filled with all
    //           weights and biases during graph creation of layers of CNN
    //fourth arg: result - vector of gradient for all weights and biases
    for(pair<string, Output> i: map_vars){
        v_weights_biases.push_back(i.second);
    }
    vector<Output> grad_outputs;
 

    TF_CHECK_OK(AddSymbolicGradients(net_root, {out_loss_var}, v_weights_biases, &grad_outputs));

    // vector<Output> loss_vector, gradient_outputs;
    // auto scewl = SoftmaxCrossEntropyWithLogits(net_root, out_classification, input_labels_var);
    // loss_vector = {scewl.loss};
    // gradient_outputs = {scewl.backprop};

    //applying Adam optimization algorithm with calculated gradient for each weight/bias
    int index = 0;  
    for(pair<string, Output> i: map_vars)
    {
        string string_index = to_string(index);
        //variable names as are used in Adam algorithm, have to be Variable()
        auto m_var = Variable(net_root, map_shapes[i.first], DT_FLOAT);
        auto v_var = Variable(net_root, map_shapes[i.first], DT_FLOAT);

        map_assigns["m_assign"+string_index] = Assign(net_root, m_var, Input::Initializer(0.f, map_shapes[i.first]));
        map_assigns["v_assign"+string_index] = Assign(net_root, v_var, Input::Initializer(0.f, map_shapes[i.first]));
        //passing a lot of values for specific elements in algorithm; can be played around with for differing results
        //most important one it seems is learning_rate, which is why it is passed through an argument in our CreateOptimizationGraph
        auto adam = ApplyAdam(net_root, i.second, m_var, v_var, 0.f, 0.f, learning_rate, 0.9f, 0.999f, 0.00000001f, grad_outputs[index]);
        v_out_grads.push_back(adam.operation);
        index++;
    }
    return net_root.status();

}

//function which runs initializating operations in a session for the whole CNN
Status CNN::Initialize()
{
    if(!net_root.ok())
        return net_root.status();
    //get all operations to run in format suitable for ClientSession's run - vector
    //and run it on train scope
    vector<Output> ops_to_run;
    for (auto i: map_assigns)
        ops_to_run.push_back(i.second);
    train_session = unique_ptr<ClientSession>(new ClientSession(net_root));
    TF_CHECK_OK(train_session->Run(ops_to_run, nullptr));

    return Status::OK();
}

//function which runs image labelling in training mode - with training elements - dropout, backpropagation and with loss evaluation
//first and second arg: tensor with a batch of images and a tensor with labels for each element in batch, already loaded
//third arg: results holds a vector of 0s and 1s where 1 means correctly classified image (> 0.5 for label in the last layer of network)
//fourth arg: one float value, calculated loss - mean squared difference
Status CNN::TrainCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results, float& loss)
{
    if(!net_root.ok())
        return net_root.status();
    
    vector<Tensor> out_tensors;
    //Args: batch of images and labels passed through arguments in this function TrainCNN which specify inputs
    //Dropout parameters: drop rate of a half and skip drop to 0, i.e. not to skip - see dropout comments in CreateGraphForCNN function
    //Backpropagation values for weights and bias from ApplyAdam function, v_out_grads 
    //Returns: Loss and result in out_tensors
    TF_CHECK_OK(train_session->Run({{input_batch_var, image_batch}, {input_labels_var, label_batch}, {drop_rate_var, 0.5f}, {skip_drop_var, 0.f}},
                                {out_loss_var, out_classification}, v_out_grads, &out_tensors));
    
    //extracting scalar values from tensors
    loss = out_tensors[0].scalar<float>()(0);
    //both labels and results are shaped [batch_size, 1]
    auto mat1 = label_batch.matrix<float>();
    auto mat2 = out_tensors[1].matrix<float>();
    //for each image push to results vector: 1 for correct classification, 0 for wrong
    //(if difference between network answer and label is higher that a half, it is incorrectly classified)
    for(int i = 0; i < mat1.dimension(0); i++)
        results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
    return Status::OK();
}

//function which runs image labelling in validation mode - without training elements - dropout, backpropagation and without loss evaluation
//more info in TrainCNN comments
Status CNN::ValidateCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results)
{
    if(!net_root.ok())
        return net_root.status();
    
    vector<Tensor> out_tensors;
    //Inputs: batch of images, drop rate 1 and skip_drop_var to 1 (which means skipping dropout phase).
    //no need to pass labels since loss isn't calculated and binary results can be easily calculated down here 
    TF_CHECK_OK(train_session->Run({{input_batch_var, image_batch}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f}}, {out_classification}, &out_tensors));
    auto mat1 = label_batch.matrix<float>();
    auto mat2 = out_tensors[0].matrix<float>();
    for(int i = 0; i < mat1.dimension(0); i++)
        results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
    return Status::OK();
}

//function which runs image labelling in prediction mode - without training elements - dropout, backpropagation and without loss evaluation
//no batches, only single images
Status CNN::Predict(Tensor& image, int& result)
{
    if(!net_root.ok())
        return net_root.status();
    
    vector<Tensor> out_tensors;
    //Inputs: image, drop rate 1 and skip drop.
    TF_CHECK_OK(train_session->Run({{input_batch_var, image}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f}}, {out_classification}, &out_tensors));
    auto mat = out_tensors[0].matrix<float>();
    result = (mat(0, 0) > 0.5f)? 1 : 0;
    return Status::OK();
}


//utility function for writing graph visualisations for TensorBoard to watch
//to see resulting graph:
//  command: tensorboard --logdir ~/tensorflow/tensorflow/examples/CNN/graphs/
//  open link in browser: http://localhost:6006/#graphs&run=
Status CNN::writeGraphForTensorboard(Scope scope, string s)
{
    //defining graph variable
    GraphDef graph;
    //extract graph object from scope using utility function ToGraphDef
    TF_RETURN_IF_ERROR(scope.ToGraphDef(&graph));
    //creating SummaryFileWriter, which will write graph files
    //first arg: max queue of graphs before writing, in this case 1
    //second arg: wait time in milliseconds, in this case 0
    //third arg: path to folder where files will be stored
    //fourth arg: file name suffix
    //sixth arg: resulting file writer
    SummaryWriterInterface* w;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/home/mordoksieznik/tensorflow/tensorflow/examples/CNN/graphs",
                "." + s + "-graph", Env::Default(), &w));
    //using created writer to write said graph
    TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
    return Status::OK();
}

int main(int argc, const char * argv[])
{
    //Creating our object and a graph of operations for loading batches of images to vectors
    int image_side = 150;
    CNN model(image_side, image_side);
    cout << "Object initialized" << endl;
    Status s;
    s = model.CreateGraphForImage(true);
    cout << "Graph for image created" << endl;
    TF_CHECK_OK(s);


    //Reading a batch of images used for training and validation from specified paths - performing operations from the graph created above
    string base_folder = "/home/mordoksieznik/tensorflow/tensorflow/examples/CNN/data/train";
    int batch_size = 20;
    vector<Tensor> image_batches, label_batches, valid_images, valid_labels;
    //Label: cat=0, dog=1
    s = model.ReadBatches(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, batch_size, image_batches, label_batches);
    cout << "Batches for train read" << endl;
    TF_CHECK_OK(s);
    
    base_folder = "/home/mordoksieznik/tensorflow/tensorflow/examples/CNN/data/validation";
    s = model.ReadBatches(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, batch_size, valid_images, valid_labels);
    cout << "Batches for validation read" << endl;
    TF_CHECK_OK(s);


    //CNN model - graph creation (without backpropagation)
    int filter_side = 3;
    s = model.CreateGraphForCNN(filter_side, filter_side);
    cout << "CNN graph for going forward created" << endl;
    TF_CHECK_OK(s);
    //backpropagation graph creation
    s = model.CreateOptimizationGraph(0.0002f);
    cout << "CNN graph with optimization created" << endl;
    TF_CHECK_OK(s);


    //uncomment to create a graph visualisation using TensorBoard 
    model.writeGraphForTensorboard(model.net_root, "cnn");
    cout << "Graph for Tensorboard drawn" << endl;


    //initializing weights and biases
    s = model.Initialize();
    cout << "Model Initialized" << endl;
    TF_CHECK_OK(s);

    //getting amount of batches for loop stopping and making sure there is the same amount of label and image batches
    size_t num_batches = image_batches.size();
    assert(num_batches == label_batches.size());
    size_t valid_batches = valid_images.size();
    assert(valid_batches == valid_labels.size());


    //Running training and validation for specified number (num_epochs) of epochs - one epoch is when the net has classified all training data once
    int num_epochs = 20;
    //Epoch / Step loops
    for(int epoch = 0; epoch < num_epochs; epoch++)
    {
        cout << "Epoch " << epoch+1 << "/" << num_epochs << ":";
        //Clock to display how long an epoch took
        auto t1 = high_resolution_clock::now();
        float loss_sum = 0;
        float accuracy_sum = 0;
        //running an epoch - looping through all batches; adding all losses and accurcies together to display average at the end of every epoch
        for(int b = 0; b < num_batches; b++)
        {
            vector<float> results;
            float loss;
            s = model.TrainCNN(image_batches[b], label_batches[b], results, loss);
            loss_sum += loss;
            accuracy_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
            cout << "." << flush;
        }

        cout << endl << "Validation:";
        float validation_sum = 0;
        for(int c = 0; c < valid_batches; c++)
        {
            vector<float> results;
            s = model.ValidateCNN(valid_images[c], valid_labels[c], results);
            validation_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
            cout << "." << flush;

        }
        auto t2 = high_resolution_clock::now();
        cout << endl << "Time: " << duration_cast<seconds>(t2-t1).count() << " seconds ";
        cout << "Loss: " << loss_sum/num_batches << " Results accuracy: " << accuracy_sum/num_batches << " Validation accuracy: " << validation_sum/valid_batches << endl;
    }

    //testing model after training - showing new images, getting results
    //rebuild the model without unstacking
    s = model.CreateGraphForImage(false);
    TF_CHECK_OK(s);
    base_folder = "/home/mordoksieznik/tensorflow/tensorflow/examples/CNN/data/test";
    vector<pair<Tensor, float>> all_files_tensors;
    s = model.ReadFileTensors(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, all_files_tensors);
    TF_CHECK_OK(s);
    //test a few images
    int count_images = 20;
    int count_success = 0;
    for(int i = 0; i < count_images; i++)
    {
        pair<Tensor, float> p = all_files_tensors[i];
        int result;
        s = model.Predict(p.first, result);
        TF_CHECK_OK(s);
        cout << "Test number: " << i + 1 << " predicted: " << result << " actual is: " << p.second << endl;
        if(result == (int)p.second)
            count_success++;
    }
    cout << "total successes: " << count_success << " out of " << count_images << endl;

    return 0;
}
