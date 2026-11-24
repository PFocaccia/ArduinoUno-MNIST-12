# Third-party notices

## MNIST-12 INT8 model

The parameters in `cnn/weights.h` were extracted from [`mnist-12-int8.onnx` in the ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist). They were converted to C++ flash constants, and the second convolution's weights were rearranged to make input-channel weights contiguous. The quantized export is credited upstream to mengniwang95, airMeng, hshen14, and the other contributors listed in the model card.

The [model card in the original ONNX repository](https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/README.md#license) identifies the model license as MIT. The parent repository uses Apache-2.0, and the mirrored model card on Hugging Face also carries an Apache-2.0 metadata tag alongside its MIT license section.

## MNIST test data

`server/mnist-images.bin` and `server/mnist-labels.bin` are derived from the MNIST database of handwritten digits, copyright Yann LeCun and Corinna Cortes. The source files are the [MNIST test images and test labels hosted by CVDF](https://github.com/cvdfoundation/mnist#download). This copy contains the first 1,000 test samples in their original order, with the IDX headers removed and the original pixels and labels preserved.

The MNIST files are distributed under [Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/), as documented by [Keras](https://keras.io/api/datasets/mnist/).
