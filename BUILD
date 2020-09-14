load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "my_project",
    srcs = ["CNN.cc", "CNN.h"],
    deps = ["//tensorflow/core:tensorflow",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/cc:client_session",
            "//tensorflow/core/kernels:image_ops",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc"

    ]
)
