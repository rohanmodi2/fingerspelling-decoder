## TFLite Model Description

### Input
```
Name: serving_default_args_0:0
Index: 0
Shape: (1, 40, 20)
Shape Signature: (-1, 40, 20)
Data Type: float32
Quantization: None
  Quantization Parameters:
    Scales: []
    Zero Points: []
```
### Output
```
Name: PartitionedCall_1:0
Index: 44
Shape: (1)
Shape Signature: (-1)
Data Type: int32
Quantization: None
  Quantization Parameters**:
    Scales: []
    Zero Points: []
```
#### Example Input

```python
import numpy as np

input_data = np.array([
    [-9.768683, -1.408827, -32.893133, -1.881516, -16.548067, -13.035077, -35.332716, -3.656107, -10.148346, -11.897635, -33.727974, -1.82994, -3.963988, -9.03396, -18.55509, 0.689644, 1.667462, -5.292356, -13.676727, 2.918041],
    [-7.621622, -1.429474, -25.770423, -5.86918, -9.614268, -15.334731, -25.741139, -6.92414, -1.913288, -14.645177, -19.183737, -3.271055, 4.967266, -11.757976, -8.292437, -0.703639, 10.681602, -7.500714, -0.577232, 0.492102],
    # ... (more input vectors as needed)
], dtype=np.float32).reshape(1, 40, 20)
```

#### Example output

```
[184, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 186, 163, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164]
```
