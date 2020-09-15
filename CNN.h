#include <iostream>
#include <map>
#include <fstream>
#include <chrono>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class CNN
{
    private:
        const int image_channels=3, image_height, image_width, image_mean, image_std;
        Output image_tensor_var, file_name_var;

        map<string, Output> map_vars, map_assigns;
        map<string, TensorShape> map_shapes;
        Output input_batch_var, drop_rate_var, skip_drop_var, out_classification;

        Output input_labels_var, out_loss_var;
        vector<Output> v_weights_biases;
        vector<Operation> v_out_grads;
        unique_ptr<ClientSession> train_session;

    public:
        Scope image_root, net_root, train_root;

        CNN(int h, int w, int mean=0, int s=255) : image_root(Scope::NewRootScope()), net_root(Scope::NewRootScope()), train_root(Scope::NewRootScope()),
                                                   image_height(h), image_width(w), image_mean(mean), image_std(s){};
        Status CreateGraphForImage(bool unstack);
        Status ReadTensorFromImageFile(const string& file_name, Tensor& out_tensor);
        Status ReadFileTensors(string& base_folder_name, vector<pair<string, float>> v_folder_label, vector<pair<Tensor, float>>& file_tensors);
        Status ReadBatches(string& base_folder_name, vector<pair<string, float>> v_folder_label, int batch_size,
                           vector<Tensor>& image_batches, vector<Tensor>& label_batches);
        
        Input AddConvolutionLayer(string index, Scope scope, int in_channels, int out_channels, int filter_height, int filter_width, Input input);
        Input AddDenseLayer(string index, Scope scope, int in_units, int out_units, bool bActivation, Input input);
        Input XavierInitialization(Scope scope, int in_channels, int out_channels, int filter_height=0, int filter_width=0);
        Status CreateGraphForCNN(int filter_height, int filter_width);
        Status CreateOptimizationGraph(float learning_rate);
        Status Initialize();
        Status TrainCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results, float& loss);
        Status ValidateCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results);
        Status Predict(Tensor& image, int& result);

        
        Status writeGraphForTensorboard(Scope scope, string s);
};

