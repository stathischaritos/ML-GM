Inference in Graphical Models
=============================
In this assignment, we will implement the sum-product and max-sum algorithms for factor graphs over discrete variables.
The relevant theory is covered in chapter 8 of Bishop's PRML book, in particular section 8.4. Read this chapter carefuly
before continuing!
We will first implement sum-product and max-sum and apply it to a simple poly-tree structured factor graph for medical
diagnosis. Then, we will implement a loopy version of the algorithms and use it for image denoising.
For this assignment we recommended you stick to numpy ndarrays (constructed with np.array, np.zeros, np.ones, etc.) as
opposed to numpy matrices, because arrays can store n-dimensional arrays whereas matrices only work for 2d arrays. We
need n-dimensional arrays in order to store conditional distributions with more than 1 conditioning variable. If you want to
perform matrix multiplication on arrays, use the np.dot function; all infix operators including *, +, -, work element-wise on
arrays.