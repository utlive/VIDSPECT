VIDSPECT Software release.

========================================================================

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2018 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
at the University of Texas at Austin (UT Austin, http://www.utexas.edu), is acknowledged in any publication 
that reports research using this code. The research is to be cited in the bibliography as:

1) T. Goodall and A. C. Bovik, "VIDSPECT Software Release", 
URL: http://live.ece.utexas.edu/research/quality/VIDSPECT_release.zip, 2018

2) T. Goodall and A. C. Bovik, "Detecting Source Video Artifacts with Supervised Sparse Filters" PCS 2018

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

Author  : Todd Goodall
Version : 1.0

The authors are with the Laboratory for Image and Video Engineering
(LIVE), Department of Electrical and Computer Engineering, The
University of Texas at Austin, Austin, TX.

Kindly report any suggestions or corrections to tgoodall@utexas.edu

========================================================================

This is a demonstration of the Video Impairment Detection by Sparse Error CapTure algorithm.
The algorithm is described in:

T. Goodall and A. C. Bovik, "Detecting Source Video Artifacts with Supervised Sparse Filters"

You can change this program as you like and use it anywhere, but please
refer to its original source (cite our paper and our web page at
http://live.ece.utexas.edu/research/quality/VIDSPECT_release.zip).

========================================================================

Running using Python

Since the data used in the original paper is not available to the public,
we constructed a few toy examples.

In a terminal, you can run:

  $ python2 VIDSPECT_patches.py 1 1 10 crosses

This will generate the data in memory, output information related to solving the 
optimization problem, then provide final output in a new "CrossesSolution" subfolder. 
This final output includes the basis functions in png form for visualization along
with a .pkl data file that can be loaded using joblib from sklearn.externals.

To generate basis functions related to identifying upscaling interpolation type:

  $ python2 VIDSPECT_patches.py 1 1 100 upscaling

Also, to generate basis functions related to the combing problem:

  $ python2 VIDSPECT_patches.py 1 1 100 combing

Note, that the basis functions used in the paper are different, but easily attainable
from real labeled video data. Also note that the classification results observed in the paper
are based on convolutions of patches with larger frames.

========================================================================
