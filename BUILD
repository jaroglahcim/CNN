load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "my_project",
    srcs = ["main.cc", "CNN.cc"],
    hdrs = ["CNN.h"],
    deps = ["//tensorflow/core:tensorflow",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/cc:client_session",
            "//tensorflow/cc:image_ops",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc"

    ]
)
