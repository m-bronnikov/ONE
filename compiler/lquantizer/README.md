# lquantizer

*lquantizer* is a tool to perform low-bit post-training quantization.

## Usage

This will run with the path to the input model (.circle), a pack of input data (.h5), path to output model (.circle) 
and encoding per parameter (weights and activations).

```
$ ./lquantizer --input_model <path_to_input_model> --output_model <path_to_output_model> \
               --input_data <path_to_input_data> --encode_bits <bits_per_encoding>
```

For example,
```
$ ./lquantizer --input_model input.circle  --output_model output.circle --input_data train.hdf5 --encode_bits 3
```

Output is a circle model where all FullyConnected nodes replaced to their corresponding LQ prototypes.
