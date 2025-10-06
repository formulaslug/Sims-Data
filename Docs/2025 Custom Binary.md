# All about the .fsdaq binary format

1. [History](#history)
1. [How it works](#hiw)
1. [Misc Details](#md) 
    1. [Why n must be a multiple of 8](#0)
    1. [Why support just these data types?](#1)

<h2 id="history"> History </h2>

When we started thinking about how to log data to an SD card, two main formats came to mind. First is ```CSV```, standard, simple. The second was using ```Arrow``` (part of the implementation of polars and how the Visualization website transfers data). ```CSV``` is super bulky and could not fit our limited write speed. ```Arrow``` was more promising, but could not be contained within the memory limits of the STM32F446RE. So we made our own method.

<h2 id="hiw"> How it works </h2>

First of all it is binary. It is modeled on how Arrow has a header followed by blocks of data called record batches. We first stick a header at the top written in ascii ```fsdaq001``` (8 bytes) followed by two 32 bit unsigned integers ```m``` and ```n``` respectively. The ```001``` that follows makes it a round 8 bytes and allows us to make new versions of the protocol and a single decoder that can support all of them!
- ```m``` is the number of columns
- ```n``` is the number of rows in a record batch and must be a multiple of 8 to support booleans. [If you want to read more about this..](#0)


After these first 16 bytes, there is a list of ```m``` column titles. Every title is a length (uint8) followed by the column title of that length. There are ```m``` of these so it checks the first length, grabs the title, then the next length, etc. These are used to generate a list of column titles.

Following that, is a list of ```m``` data types. These are a character (f, i, u, b, for float, int, unsigned int, and bool respecitvely) followed by an integer (ascii) that is the power of 2 the data type takes up in bits. Some examples:
- ```b0``` is a bool for ```2^0 = 1 bit```
- ```u5``` is a 32 bit unsigned integer
- ```f5``` is a 32 bit float

currently only these are supported. [Why are just these supported?](#1)
    
```python
"i3": np.int8
"i4": np.int16
"i5": np.int32
"i6": np.int64
"u3": np.uint8
"u4": np.uint16
"u5": np.uint32
"u6": np.uint64
"f4": np.float16
"f5": np.float32
"f6": np.float64
"b0": bool
```

Ok, now we have our column titles and data types. After this, each data chunk is read in. For every column you grab ```n * <data size>``` and pull out ```n``` values that get stored as a polars ```Series```. You join every single series to form a polars ```DataFrame```. You create a list of these for every record batch and then concatenate them all at the end to form a single ```DataFrame``` that gets written to a ```.parquet``` file specified! And that's it :)

<h2 id="md"> Misc Details </h2>

<h3 id="0"> Why n must be a multiple of 8 </h3>

When packing booleans into bytes you can either pad every boolean with 7 0s to make a byte, pad ```n``` booleans into some number of bytes with a single byte on the end that is padded, or set ```n``` as a multiple of 8 making every set of booleans fit in a byte.

<h3 id="1"> Why support just these data types? </h3>

Our communication protocol on the car is CAN and you can only fit 8 bytes into a single message (64 bits). It is unwieldy and impractical to send a single piece of information across multiple CAN messages because they get lost fairly frequently. Thus, we will not support precision greater than 64 bits.

Floats under 16 bits hardly have any precision so we don't support those.

Ints under a byte have little use for us so unless a special case arises, we will likely not support this either.
