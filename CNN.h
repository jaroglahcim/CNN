#include <iostream>
#include <map>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class CNN
{
    private:
        Scope image_root;
        const int image_channels=3, image_height, image_width, image_mean, image_std;
        Output image_tensor_var, file_name_var;

        map<string, Output> map_vars, map_assigns;
        map<string, TensorShape> map_shapes;
        Output input_batch_var, drop_rate_var, skip_drop_var, out_classification;
        Scope net_root;

    public:
        CNN(int h, int w, int mean=0, int s=255) : image_root(Scope::NewRootScope()), image_height(h), image_width(w), image_mean(mean), image_std(s){};
        Status CreateGraphForImage(bool unstack);
        Status ReadTensorFromImageFile(const string& file_name, Tensor& out_tensor);
        Status ReadFileTensors(string& base_folder_name, vector<pair<string, float>> v_folder_label, vector<pair<Tensor, float>>& file_tensors);
        Status ReadBatches(string& base_folder_name, vector<pair<string, float>> v_folder_label, int batch_size,
                           vector<Tensor>& image_batches, vector<Tensor>& label_batches);
        
        Input AddConvolutionLayer(string index, Scope scope, int in_channels, int out_channels, int filter_height, int filter_width, Input input);
        Input XavierInitialization(Scope scope, int in_channels, int out_channels, int filter_height=0, int filter_width=0);
        Input AddDenseLayer(string index, Scope scope, int in_units, int out_units, bool bActivation, Input input);
        Status CreateGraphForCNN(int filter_height, int filter_width);
};

