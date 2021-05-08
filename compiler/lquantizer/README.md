# lquantizer

_lquantizer_ is a tool to perform LQ post-training quantization.

## Usage

This will run with the path to the input model (.circle), a pack of input data (.h5), and the output model (.circle).

```
$ ./lquantizer <path_to_input_model> <path_to_input_data> <path_to_output_model>
```

For example,
```
$ ./lquantizer input.circle input.h5 out.circle
```

Output is a circle model where some nodes replaced to their corresponding LQ prototypes.
