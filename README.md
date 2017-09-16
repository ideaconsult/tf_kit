# Tensor Flow misc Kit
A small collection of python scripts for running neural-net models with [TensorFlow](https://www.tensorflow.org). All scripts are both Python 2.7 and Python 3.6 tested and compatible with TF versions 1.0 and 0.12. Also all runnable scripts can be invoked with `-h` argument to get help information about them.

## <a name="building">Model building scripts</a>
The paradigm behind organizing the scripts is to separate the model creation process from training and from inference runs of the trained model. Following that idea the two implemented types of models - _(Denoising) Auto-Encoders_ and _Variational Auto-Encoders_ have the respectful scripts `tf_ae.py` and `tf_vae.py` which only purpose is to build the architecture given in the arguments. This is not a time consuming process. A call may look like this:

```
$ tf_vae.py 773x9x1:c,1x9x3,relu,1x1,valid:d,1200,relu:200,sigmoid -l 0.001 -b 200 -c xentropy -m hci_vae/model.meta
```

The first and most important argument is the architecture specification. In the case of autoencoders, this should only specify the _encoder_ part, because the _decoder_ part is automatically built by reversing the given one. It consists of per-layer definitions separated by colon (`:`), the first layer being an exception - it defines the size and/or shape of the data samples. So, the four layer here are:

- `773x9x1`: An input shape of 773 in height, 9 in width and 1 channel in depth bunch of number. Especially useful for images. Could be only a number, meaning the size of the input, e.g. **6957**.
- `c,1x5x2,elu,1x1,valid`: A **c**onvolution layer, with filter of size **1x5** - two (**2**) of them, with **1x1** striding and **valid** padding. The activation function is `elu`. Could also be `rely`, `sigmoid`, `tanh` and  `id`.
- `d,1500,elu`: A **d**ense layer - fully connected from the output of the previous one, with **1500** neurons and `elu` activation function - the possible function names are the same as for the convolution layers.
- `200,id`: The last layer determines the latent space - **200** in this case, with the output activation function. In the case of auto-encoders this activation function is used at the very end of the decoder, i.e. after the symmetrical part of the network is build, at the last step of building again the input-space sized layer for decoded data.

The rest of the parameters to the building script include the `-b`atch size, because it can't always be deduces, the `-c`ost function (`mse`, `xentropy`), the `-l`earning rate, because usually the optimizer is quite model dependent, and finally - where the created `-m`odel should be saved. In order for the training script to take over, it is important that certain naming / rules are followed while the model is created:
- The important ops should be added to specifically named collections: `input_var`, `latent_var`, `generative_var`, `output_var`, `train_op` and `loss_op`.
- Additionally the placeholder for the model input should be named `input_pipe`, because upon model restoration it is replaced with the actual data input pipe.
- For clarification - the `output_var` is the decoded counterpart of the input, the `latent_var` is the encoded version, prior any artificial noise or variance, and `generative_var` is the one that need to be fed in order to get newly generated samples, i.e. it is _after_ the noise / variation steps and thus feeding it overrides them. `input_var` and `output_var` have same dimensionality and so does `latent_var` and `generative_var`.
- It is important that these are added in collection, so they'll be retrieved like this: `train_op = tf.get_collection("train_ops")[0]`, for example. 

At the heart of both scripts is corresponding Python class, which can freely be used as part of another environment. All those key ops added to the collections as described above, are also exposed as properties, so the model classes can be accessed in a unified way.

## <a name="training">`tf_train.py` - model training script</a>

The concept for this tool is to be flexible, model-invariant script that can train any model, which exposes properly needed tensors (see above) It basically does that:
* Establishing the TensorFlow reading pipeline
* Loads the specified model, replacing it's input with the established pipeline.
* If asked, prepares a validation set and adds necessary hooks for performing validation regularly.
* Establishes the hooks for saving checkpoints of the trained model, so it can later be retrieved.

An example of using the script is:

```
$ tf_train.py -m hci_vae/model.meta -t data/ordered_train.txt -f csv_mem -d ' ' -v data/ordered_test.txt -y 5 -e 100 -b 200
```

The first argument is again a reference to where the `-m`odel was saved with some of model building scripts [see above](#building). The follows the `-t`est data location, the `-f`ormat of the data, the `-d`elimiter to be expected. The `-v`alidation set can also be specified and if possible it is better to use `--validation-format csv_mem` which will preload the validation set in memory and make validation steps faster. Earl`-y` stopping can also be utilized, providing after how many validation steps of non-improving loss the process should be stopped. Another stopping criteria, of course, is the `-e`pochs number. 

Note, that the `-b`atch size should be specified again and it should be the same as the one found in the model. Actually the batch specification belongs more to _this_ script, but the nature to some models, like variational auto-encoders and denoising auto-encoders require knowing the batch size in advance.

Another _quite useful_ and important feature is that training can be restarted just by providing the location of the _model_, like this:

```
$ tf_train.py -m hci_vae/model.meta -v data/ordered_test.txt -y 5
```

Pay attention, that _validation_ data should be provided again (or it can be different), as well as the _early stopping steps_. On the other hand, neither the _training data_ (along with the _format_ and _delimiter_), nor the number of training _epochs_ can be altered, because they have become part of the input pipe, i.e. - from the TF calculation graph.

## `tf_run.py` - using the trained model

After the model training has finished, its usage is bundled in this script, which has even lighter syntax, like:

```
$ tf_run.py -m hci_vae/model.meta -t data/ordered_small.txt -f csv_mem -d " " -o hci_ae.txt
```

Again the trained `-m`odel location is provided, the `-t`est data and it's `-f`ormat and `-d`elimiter. The `-o`utput can either be specified or expected on the standard output.

## `tf_utils.py` - a set of TensorFlow convenient methods

The actual architecture building routines reside here, which take a parsed architecture like the one, explained before, and construct the actual TF ops and tensors graph. The main function for that is `tf_build_architecture`, but it relies on `tf_conv_layer` and `tf_dense_layer` to do the actual work.

Another set of helpful functions is concerning the input, they are aimed at big dataset, so the recommended Queue-based approach is utilized combining properly several functions: `tf_file_pipe`,  `tf_csv_reader` and `tf_pipe_shuffle`. For direct, in-memory reading of smaller dataset there is still the `tf_const_input`, which can be combined with `tf.train.input_producer` (standard in TensorFlow implementation) and `tf_pipe_shuffle`.


## `tf_persist.py` - the model persistency tools

Saving a model, after it has been created and restoring it for the training and running processes is achieved by routines implemented in this script. Namely - `tf_export_graph` and `tf_restore_graph`. Also there is a very lightweight class, responsible for holding the retrieved key ops for easier (and coherent with actual model classes from other scripts) access, which is named `RestoredModel`.

## `nn_utils.py` - a general NN tools

The most important one, being the `nn_parse_architecture(arch_str, func_dict)` one. The input of the architecture was described above, the corresponding list, which will be returned and will usable by the model initializers would look like this:

```
[
	{ 'sample_shape': [773,9,1] }, 
	{ 'type': "conv", 'filter_shape': [1,5], 'strides': [1,1], 'filters': 2, 'padding': "valid" , 'func': "flu"},
	{ 'type': "dense", 'size': 1500, 'func': "elu" },
	{ 'size': 500, 'func':  "id" }
]
```

A couple useful functions concern reading of MNIST set `packed_images_reader`, including the downloading process. There are plans to put into same context management of CIFAR dataset, for easier experiments.

## Another possible runs and tests

Using the script above to create and train a VAE model for MNIST set:

```
$ tf_vae.py 28x28x1:c,5x5x16,relu,valid,2x2:c,5x5x32,relu,same,2x2:50,sigmoid -b 200 -l 0.001 -m mnist_vae/model.meta
$ tf_train.py -m mnist_vae/model.meta -b 200 -d train-images-idx3-ubyte.gz -f images -v test-images-idx3-ubyte.gz -y 10
```

Running a non-convolution VAE on the raw (merged) HCI data, but with normalized values (in the range [0, 1]), would look like this:

```
$ tf_vae.py 6957:d,2500:d,1500:500,sigmoid -b 500 -l 0.0001 -c xentropy -m hci_vae_flat/model.meta
$ tf_train.py -m hci_vae_flat/model.meta -b 500 -t data/norm_train.txt -d " " -f csv -v data/norm_test.txt -c 600 -s 100 -y 10 -e 1000
$ tf_run.py -m hci_vae_flat/model.meta -t data/normalized.txt -d " " > hci_reduced.txt
```
