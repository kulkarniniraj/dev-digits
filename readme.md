## Devanagari digits dataset for AI training
- Total 17230 images for 0-9 digits
- Images for each digit are located in hdf['digit']
- Each image is of dimensions 32x32
- My code to load dataset:
```python
y = []
x = []

df = h5py.File('ocr_dev_digits.hdf5', 'r')
for i in range(10):
  x.append(df[`i`])
  y.append(np.full((df[`i`].shape[0], 1), i))
X = np.vstack(x)
Y = np.vstack(y)
print X.shape, Y.shape
# (17230, 32, 32) (17230, 1)
```

A demo of convnet trained on this data is available at 
- <https://kulkarniniraj.github.io/ocr.html> (for desktop) 
- <https://kulkarniniraj.github.io/mocr.html> (for mobile)