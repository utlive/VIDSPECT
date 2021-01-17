import os
import sys
from scipy import linalg
import numpy as np
import scipy
from sklearn.externals import joblib
import skimage.io
import skvideo.utils
import skvideo.io
import skvideo.datasets
import sklearn.svm

def plot_weights(basis):
    n_filters, n_channels, height, width = basis.shape

    if n_filters == 676:
      ncols = 26
      nrows = 26
    elif n_filters == 1352:
      ncols = 26
      nrows = 52
    elif n_filters == 338:
      ncols = 26
      nrows = 13
    elif n_filters == 200:
      ncols = 20
      nrows = 10
    elif n_filters == 100:
      ncols = 10
      nrows = 10
    elif n_filters == 40:
      ncols = 5
      nrows = 8
    elif n_filters == 20:
      ncols = 5
      nrows = 4
    elif n_filters == 10:
      ncols = 5
      nrows = 2
    elif n_filters < 10:
      ncols = n_filters
      nrows = 1
    else:
      print("unknown")
      exit(0)

    # white lines
    data = np.ones((ncols*26, nrows*26))*255
    rown = 0
    coln = 0
    for f in xrange(n_filters):
      sp = basis[f, 0, :, :]
      mi = np.min(sp)
      ma = np.max(sp)
      ma = np.max((np.abs(mi), np.abs(ma)))
      sp = sp / ma
      sp *= 127.5
      data[(coln*26):(coln*26+25), (rown*26):(rown*26+25)] = sp+127.5

      coln += 1
      if coln >= ncols:
          coln = 0
          rown += 1
    return data.astype(np.uint8)

# positive-only sparsity constraint
def shrinkage_pos(x, kappa):
  return np.maximum( 0, x - kappa )

def lassoADMMDictCodeSolver(X, g, params, ext, process_lambda, alpha_weight, modify_dictionary, MAXITER=200):
  Derror = 1e9
  ABSTOL = 0.0001

  n_filters = ext[2].shape[0]

  ext_subprob = []

  W = params[0]
  W *= 0

  mask = np.zeros(W.shape)
  # enforce positive correlations
  cls = n_filters/2
  n_cls = cls/g.shape[1]
  for i in xrange(g.shape[1]-1):
    mask[cls + i*n_cls:cls + (i+1)*n_cls, i] = 1
  mask[cls + (g.shape[1]-1)*n_cls:, g.shape[1]-1] = 1

  # for masking responses that lie outside of allowed range
  Dmask = np.zeros(ext[0].shape)
  Dmask[:cls, :] = 1
  for i in xrange(g.shape[1]-1):
    Dmask[cls + i*n_cls:cls + (i+1)*n_cls, g[:, i]==1] = 1
  Dmask[cls + (g.shape[1]-1)*n_cls:, g[:, (g.shape[1]-1)]==1] = 1

  #mask[cls + 1*n_cls:cls + 2*n_cls, 1] = 1
  #mask[cls + 2*n_cls:cls + 3*n_cls, 2] = 1
  #mask[cls + 3*n_cls:cls + 4*n_cls, 3] = 1

  numElems = W.shape[0]*W.shape[1]
  bnds = []
  for i in xrange(numElems):
    mr = mask.ravel()
    if mr[i] == 0:
      bnds.append((0, 0))
    else:
      bnds.append((0, None))

  # for the bias, only allow negatives for bias against, so correlation is for 
  # overcoming bias
  for i in xrange(g.shape[1]):
    bnds.append((None, 0))

  # abs tolerance
  esp_X_pri = ABSTOL * np.sqrt(25*25 * n_filters * X.shape[0]) 
  esp_X_dual = ABSTOL * np.sqrt(25*25 * n_filters * X.shape[0])
  esp_D_pri = ABSTOL * np.sqrt(25*25 * n_filters)

  rho = 5.0
  kk = 0

  eyemat = np.eye(n_filters).astype(np.float32)

  converged = False
  last_error = -1

  cerror = 0

  c_k = 1e9
  alpha_k = 1

  primal_Xp = ext[0]
  dual_Xp = ext[1]
  D = ext[2]

  def cross_entropy_class(W, C, g):
    W = W.reshape((C.shape[0], g.shape[1]))
    fc = np.dot(W.T,C)
    maxval = np.max(fc, axis=0)
    softmax = np.exp(fc-maxval) / np.sum(np.exp(fc-maxval), axis=0)
    Waug = W.copy()
    Waug[-1, :] = 0
    return -np.sum(g.T * np.log(softmax+1e-8)) + np.sum(Waug**2)

  def cross_entropy_classdW(W, C, g):
    #C = C.reshape((W.shape[0], g.shape[0]))
    W = W.reshape((C.shape[0], g.shape[1]))
    fc = np.dot(W.T,C)
    maxval = np.max(fc, axis=0)
    softmax = np.exp(fc-maxval) / np.sum(np.exp(fc-maxval), axis=0)

    Waug = W.copy()
    Waug[-1, :] = 0
    return np.ravel(np.dot(softmax - g.T, C.T).T + 2 * Waug)

  def L(C, D, W, bias, X, g, primal, dual, rho, alpha):
    C = C.reshape((D.shape[0], X.shape[0])).astype(np.float32)

    err = 0.5*np.sum((X.T - np.dot(D.T, C))**2)

    # use softmax to compute probabilities for each class
    Wstar = np.vstack((W, bias))
    fc = np.dot(Wstar.T,np.vstack((C, np.ones((1, C.shape[1])))))
    maxval = np.max(fc, axis=0)
    softmax = np.exp(fc-maxval) / np.sum(np.exp(fc-maxval), axis=0)
    err += -alpha*np.sum(g.T * np.log(softmax+1e-8))

    err += 0.5*rho*np.sum((C - primal + dual/rho)**2)

    return err

  def dLdX(C, D, W, bias, X, g, primal, dual, rho, alpha):
    C = C.reshape((D.shape[0], X.shape[0])).astype(np.float32)
    dX = -np.dot(D, (X.T - np.dot(D.T, C)))

    Wstar = np.vstack((W, bias))

    fc = np.dot(Wstar.T,np.vstack((C, np.ones((1, C.shape[1])))))
    maxval = np.max(fc, axis=0)
    softmax = np.exp(fc-maxval) / np.sum(np.exp(fc-maxval), axis=0)

    # we meant to not use Wstar below, since we don't want to compute gradient for 
    # the all-ones code
    dX += alpha*np.dot(W, softmax - g.T)
    dX += C*rho
    dX += -rho* primal + dual
    return dX.ravel().astype(np.float64)

  #W = np.random.random(size=W.shape)

  bias = np.zeros(g.shape[1]).reshape(1, g.shape[1])
  C = primal_Xp.copy().astype(np.float32)
  g = np.asfortranarray(g)#.astype(np.float32)

  WstarTrue = np.vstack((W, bias))

  while True:
    kk += 1

    primal1_old = primal_Xp.copy()

    C, f, _ = scipy.optimize.fmin_l_bfgs_b(L, np.ravel(C), fprime=dLdX, args=(D, W, bias, X, g, primal_Xp, dual_Xp, rho, alpha), disp=False, maxiter=100)
    C = C.reshape(primal_Xp.shape).astype(np.float32)



    # bfgs?
    #DDTandWWT = np.dot(D, D.T) + alpha_weight * (np.dot(W, W.T))
    #DX_X = np.dot(D, X.T) + alpha_weight * np.dot(W, g.T)

    #Aterm = DDTandWWT + rho * eyemat
    #Bterm = DX_X + rho * primal_Xp - dual_Xp
    #C = np.linalg.solve(Aterm, Bterm)

    # Update dual and primal
    primal_Xp = shrinkage_pos(C + dual_Xp/rho, process_lambda/rho)
    dual_Xp = dual_Xp + rho * (C - primal_Xp)

    ext[0] = primal_Xp
    ext[1] = dual_Xp

    #del Aterm
    #del Bterm

    D_old = D.copy()
    # Update dictionary, using primal
    if alpha > 0:
      As = np.dot(Dmask*primal_Xp, (Dmask*primal_Xp).T)
      Bs = np.dot(Dmask*primal_Xp, X)
      D = np.linalg.solve(As, Bs)
    else:
      As = np.dot(primal_Xp, (primal_Xp).T)
      Bs = np.dot(primal_Xp, X)
      D = np.linalg.solve(As, Bs)

    ext[2] = D.astype(np.float32).copy()

    # constrain shape with Gaussian multiply
    for i in range(D.shape[0]):
      #D[i,:] *= kern
      norm = np.sqrt(np.dot(D[i,:], D[i,:]))
      if norm < 1e-9:
        D[i, :] = np.random.random(25**2)
        #D[i,:] *= kern
        D[i, :] /= np.sqrt(np.dot(D[i, :], D[i, :]))
      else:
        D[i, :] /= norm

    D = D.astype(np.float32)
    Derror = np.sqrt(np.sum((D - D_old)**2))

    # Update classification weights, using primal
    # we want to norm the weights on W, using l2 constraint 
    #wi = w_i-1 + 1/(2.0
    #lmbda = 1.0
    #W[:, i] += 1/(2.0 * lmbda) * wi = -1.0/lmbda * X[:, 0] * 
    ##min_W ||g - WX|| + sum_i ||w_i||_2
    #d/dw_i = X_i.T (g - W X) + w_i
    #0 = X_i.T(g-WX) + lmbda * w_i
    #w_i = - 1.0/lmbda * X_i.T * (g-WX) 

    if kk > -1:
      if np.abs(alpha_weight - 0) <= 1e-6:
        W *= 0
      else:
        #W = solveWb_nnls(primal_Xp, g)
        #W = solveWb_nnls(primal_Xp[n_filters/2:, :], g)
        #W = np.vstack((W*0, W))
        #indepCode = code_step(np.asfortranarray(X.T), np.asfortranarray(D.astype(np.float32).T), process_lambda)
        #cov = np.dot(primal_Xp, primal_Xp.T).T
        #cov *= np.eye(n_filters)
        #W = np.linalg.solve(cov, np.dot(primal_Xp, g))

        if 0:
          #exit(0)
          mask = np.zeros(W.shape)
          # enforce positive correlations
          cls = n_filters/2
          n_cls = cls/g.shape[1]
          mask[cls:cls + 1*n_cls, 0] = 1
          mask[cls + 1*n_cls:cls + 2*n_cls, 1] = 1
          mask[cls + 2*n_cls:cls + 3*n_cls, 2] = 1
          mask[cls + 3*n_cls:cls + 4*n_cls, 3] = 1
          mask[cls + 4*n_cls:, 4] = 1

          W = mask

        #  W = np.linalg.solve(np.dot(primal_Xp, primal_Xp.T).T + np.eye(n_filters), np.dot(primal_Xp, g))
        #  print "regularized W update"
          #for i in xrange(W.shape[0]):
          #  norm = np.sqrt(np.dot(W[i, :], W[i, :]))
          #  if norm > 1e-9:
          #     W[i, :] /= norm
        else:
          Wstar = np.vstack((W, bias))
          primal_Xpstar = np.vstack((primal_Xp, np.ones((1, primal_Xp.shape[1]))))
          Wstar2, f, _ = scipy.optimize.fmin_l_bfgs_b(cross_entropy_class, np.ravel(Wstar), fprime=cross_entropy_classdW, args=(primal_Xpstar, g), disp=False, maxiter=100, bounds=bnds)
          Wstar = Wstar2.reshape(Wstar.shape)
          W = Wstar[:n_filters]
          bias = Wstar[n_filters:]

          # get mask of zeros where we want otherwise  
          #invalidsmask = (mask == 1) & (W == 0)

          #print np.sum(invalidsmask), " invalids"
          #randW = np.random.random(size=invalidsmask.shape)

          #W[invalidsmask] = np.max(W)#randW[invalidsmask]
          #W = Wstar
        #else:
        #  W[i, :] = np.random.random(W.shape[1])
        #  W[i, :] /= np.sqrt(np.dot(W[i, :], W[i, :]))

    fc = np.dot(W.T, primal_Xp) + bias.T
    maxval = np.max(fc, axis=0)
    softmax = np.exp(fc-maxval) / np.sum(np.exp(fc-maxval), axis=0)
    self_error = -np.sum(g.T * np.log(softmax+1e-8))
    #self_error = np.mean((g.T - np.tanh(np.dot(W.T, primal_Xp)))**2)
    r_norm1 = np.sqrt(np.sum((C-primal_Xp)**2))
    s_norm1 = rho*np.sqrt(np.sum((primal1_old-primal_Xp)**2))

    del primal1_old
    #del C

    if ((kk % 10) == 0):
      if r_norm1 > 10*s_norm1:
        rho = 2.0*rho
      elif r_norm1*10 < s_norm1:
        rho = (0.5)*rho

      if rho > 10:
        rho = 10
    
      if rho < 1e-4:
        rho = 1e-4
      print "rho: ", rho

    if (r_norm1 < esp_X_pri) and (s_norm1 < esp_X_dual) and (Derror < esp_D_pri):
      converged = True
      break

    print("# %d, primal: %.4f, dual: %.4f, D error: %.4f" % (kk, r_norm1, s_norm1, Derror))

    if kk >= MAXITER:
      break

  # set the returns
  ext[0] = primal_Xp
  ext[1] = dual_Xp 
  ext[2] = D


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def objective(X, D, W, b, s, g, lmbda, alpha):
  #t = np.dot(D, s.T)
  innerprod = (np.dot(W.T, X) + b.reshape(b.shape[0],1))

  error = 0.5*np.sum((s.T - np.dot(D.T, X))**2)
  error += 0.5*alpha*np.sum((g.T - innerprod)**2)
  error += lmbda * np.sum(np.abs(X))

  return error

def prepData(detType):
    def makeGaussian(size, fwhm = 3, center=None):
      """ Make a square gaussian kernel.

      size is the length of a side of the square
      fwhm is full-width-half-maximum, which
      can be thought of as an effective radius.
      """

      x = np.arange(0, size, 1, float)
      y = x[:,np.newaxis]

      if center is None:
        x0 = y0 = size // 2
      else:
        x0 = center[0]
        y0 = center[1]

      return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    def plot_patches(patches, n_cols, n_rows, drawzeromean=True):
      n_basis, height, width, n_channels = patches.shape

      buff = np.ones(((height+1)*n_rows, (width+1)*n_cols)) * 255
      cidx = 0
      ridx = 0
      for i in range(n_basis):
        bfunc = patches[i, :, :, 0].astype(np.float32)
        if drawzeromean:
          bfunc /= np.max(np.abs(bfunc))
          bfunc *= 127.5
          bfunc += 127.5
        else:
          bfunc -= np.min(bfunc)
          bfunc /= np.max(bfunc)
          bfunc *= 255.0
        buff[ridx*(height+1):ridx*(height+1)+height, cidx*(width+1):cidx*(width+1)+width] = bfunc 
        ridx += 1
        if ridx >= n_rows:
          ridx = 0
          cidx += 1
      return buff.astype(np.uint8)

    kern = makeGaussian(25, 12.0/2.0).reshape(-1)

    patch_size=25
    if detType == "crosses":
      np.random.seed(0)
      saveFolder = "CrossesSolution"

      ensure_dir(saveFolder + "/")

      n_labels = 4
      n_samples = 2000
      patch_size_d = patch_size + 20*2 # buffer so we don't have zero padding during mscn
      noise_sigma = 255
      trainX = np.zeros((n_samples, patch_size_d, patch_size_d, 1))
      label_detect = np.zeros((trainX.shape[0], 4))
      for i in range(n_samples):
        if (i < n_samples*0.25):
          trainX[i, patch_size_d/2, :patch_size_d/2] = 255
          label_detect[i, 0] = 1
        elif (i < n_samples*0.5):
          trainX[i, patch_size_d/2, patch_size_d/2:] = 255
          label_detect[i, 1] = 1
        elif (i < n_samples*0.75):
          trainX[i, :patch_size_d/2, patch_size_d/2] = 255
          label_detect[i, 2] = 1
        else:
          trainX[i, patch_size_d/2:, patch_size_d/2] = 255
          label_detect[i, 3] = 1
        trainX[i] += np.random.random(size=(patch_size_d, patch_size_d, 1))*noise_sigma

      # randomize
      idx = np.random.permutation(n_samples)
      trainX = trainX[idx]
      label_detect = label_detect[idx]

      # save the trainX as an image to show
      patchRender = plot_patches(trainX[:1000, 20:-20, 20:-20] , 4, 250, drawzeromean=False)
      skimage.io.imsave(saveFolder + "/trainingCrosses.png", patchRender)

      # now normalize and window each using MSCN transform
      for i in range(n_samples):
        trainX[i, :, :, 0], _, _ =  skvideo.utils.compute_image_mscn_transform(trainX[i, :, :, 0])
      patchRender = plot_patches(trainX[:1000, 20:-20, 20:-20], 4, 250)
      skimage.io.imsave(saveFolder + "/trainingCrosses_mscn.png", patchRender)

      trainX = trainX[:, 20:-20, 20:-20]
      trainX = trainX.reshape(-1, 625)
      trainX *= kern
      patchRender = plot_patches(trainX.reshape(n_samples, patch_size, patch_size, 1)[:1000], 4, 250)
      skimage.io.imsave(saveFolder + "/trainingCrosses_mscn_windowed.png", patchRender)

      trainX = trainX.astype(np.float32)

      return saveFolder, label_detect[:1000], trainX[:1000], label_detect[1000:], trainX[1000:] 

    elif detType == "upscaling":
      import cv2
      np.random.seed(0)
      saveFolder = "UpscalingSolution"

      ensure_dir(saveFolder + "/")

      # get bigbuckbunny data and extract 1000 patches for training
      trainvideodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny(), as_grey=True)

      # non-upscaled, lanczos, nearest neighbor, bicubic, bilinear
      n_labels = 5
      patch_size_d = patch_size + 20*2 # buffer so we don't have zero padding during mscn
      n_samples = 1000
      trainX = np.zeros((n_samples, patch_size_d, patch_size_d, 1))
      trainlabel = np.zeros((trainX.shape[0], 5))
      T, M, N, C = trainvideodata.shape
      for i in range(n_samples):
        rframe = np.random.randint(0, T)
        frame = trainvideodata[rframe, :, :, 0]
        if (i < n_samples*0.20):
          trainlabel[i, 0] = 1
        elif (i < n_samples*0.40):
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_NEAREST)
          trainlabel[i, 1] = 1
        elif (i < n_samples*0.60):
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_LINEAR)
          trainlabel[i, 2] = 1
        elif (i < n_samples*0.80):
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_CUBIC)
          trainlabel[i, 3] = 1
        else:
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_LANCZOS4)
          trainlabel[i, 4] = 1

        y = np.random.randint(0, M - patch_size_d)
        x = np.random.randint(0, N - patch_size_d)
        patch = frame[y:y+patch_size_d, x:x+patch_size_d]
        trainX[i, :, :, 0] = patch

      # save the trainX as an image to show
      patchRender = plot_patches(trainX[:, 20:-20, 20:-20] , 5, 200, drawzeromean=False)
      skimage.io.imsave(saveFolder + "/trainingUpscaled.png", patchRender)

      # now normalize and window each using MSCN transform
      for i in range(n_samples):
        trainX[i, :, :, 0], _, _ =  skvideo.utils.compute_image_mscn_transform(trainX[i, :, :, 0])
      patchRender = plot_patches(trainX[:, 20:-20, 20:-20], 4, 250)
      skimage.io.imsave(saveFolder + "/trainingUpscaled_mscn.png", patchRender)

      trainX = trainX[:, 20:-20, 20:-20]
      trainX = trainX.reshape(-1, 625)
      trainX *= kern
      patchRender = plot_patches(trainX.reshape(n_samples, patch_size, patch_size, 1), 4, 250)
      skimage.io.imsave(saveFolder + "/trainingCrosses_mscn_windowed.png", patchRender)

      trainX = trainX.astype(np.float32)

      # setup test data
      testvideodata = skvideo.io.vread(skvideo.datasets.bikes(), as_grey=True)
      patch_size_d = patch_size + 20*2 # buffer so we don't have zero padding during mscn
      n_samples = 1000
      testX = np.zeros((n_samples, patch_size_d, patch_size_d, 1))
      testlabel = np.zeros((testX.shape[0], 5))
      T, M, N, C = testvideodata.shape
      for i in range(n_samples):
        rframe = np.random.randint(0, T)
        frame = testvideodata[rframe, :, :, 0]
        if (i < n_samples*0.20):
          testlabel[i, 0] = 1
        elif (i < n_samples*0.40):
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_NEAREST)
          testlabel[i, 1] = 1
        elif (i < n_samples*0.60):
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_LINEAR)
          testlabel[i, 2] = 1
        elif (i < n_samples*0.80):
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_CUBIC)
          testlabel[i, 3] = 1
        else:
          frame = cv2.resize(frame, (N/2, M/2), interpolation=cv2.INTER_LANCZOS4)
          frame = cv2.resize(frame, (N, M), interpolation=cv2.INTER_LANCZOS4)
          testlabel[i, 4] = 1

        y = np.random.randint(0, M - patch_size_d)
        x = np.random.randint(0, N - patch_size_d)
        patch = frame[y:y+patch_size_d, x:x+patch_size_d]
        testX[i, :, :, 0] = patch

      for i in range(n_samples):
        testX[i, :, :, 0], _, _ =  skvideo.utils.compute_image_mscn_transform(testX[i, :, :, 0])

      testX = testX[:, 20:-20, 20:-20]
      testX = testX.reshape(-1, 625)
      testX *= kern

      testX = testX.astype(np.float32)

      return saveFolder, trainlabel, trainX, testlabel, testX 

    elif detType == "combing":
      import cv2
      np.random.seed(0)
      saveFolder = "CombingSolution"

      ensure_dir(saveFolder + "/")

      # get bigbuckbunny data and extract 1000 patches for training
      trainvideodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny(), as_grey=True)

      # non-upscaled, lanczos, nearest neighbor, bicubic, bilinear
      n_labels = 2
      patch_size_d = patch_size + 20*2 # buffer so we don't have zero padding during mscn
      n_samples = 1000
      trainX = np.zeros((n_samples, patch_size_d, patch_size_d, 1))
      trainlabel = np.zeros((trainX.shape[0], 2))
      T, M, N, C = trainvideodata.shape
      i = 0
      while i < n_samples:
        rframe = np.random.randint(0, T-1)
        frames = trainvideodata[rframe:rframe+2, :, :, 0].astype(np.float32)
        if (i < n_samples*0.50):
          trainlabel[i, 0] = 1
          frame = frames[0].copy()
        else:
          frame = frames[0].copy()
          frame[::2] = frames[1, ::2]
          trainlabel[i, 1] = 1

        y = np.random.randint(0, M - patch_size_d)
        x = np.random.randint(0, N - patch_size_d)
        patch = frame[y:y+patch_size_d, x:x+patch_size_d]
        if trainlabel[i, 0] == 0:
          mse = np.mean(np.abs(patch[20:-20, 20:-20] - frames[0, y+20:y+patch_size_d-20, x+20:x+patch_size_d-20]))
          if mse < 1:
            continue
        trainX[i, :, :, 0] = patch
        i+=1

      # save the trainX as an image to show
      patchRender = plot_patches(trainX[:, 20:-20, 20:-20] , 2, 500, drawzeromean=False)
      skimage.io.imsave(saveFolder + "/trainingCombing.png", patchRender)

      # now normalize and window each using MSCN transform
      for i in range(n_samples):
        trainX[i, :, :, 0], _, _ =  skvideo.utils.compute_image_mscn_transform(trainX[i, :, :, 0])
      patchRender = plot_patches(trainX[:, 20:-20, 20:-20], 2, 500)
      skimage.io.imsave(saveFolder + "/trainingCombing_mscn.png", patchRender)

      trainX = trainX[:, 20:-20, 20:-20]
      trainX = trainX.reshape(-1, 625)
      trainX *= kern
      patchRender = plot_patches(trainX.reshape(n_samples, patch_size, patch_size, 1), 2, 500)
      skimage.io.imsave(saveFolder + "/trainingCombing_mscn_windowed.png", patchRender)

      trainX = trainX.astype(np.float32)

      # setup test data
      testvideodata = skvideo.io.vread(skvideo.datasets.bikes(), as_grey=True)
      n_labels = 2
      patch_size_d = patch_size + 20*2 # buffer so we don't have zero padding during mscn
      n_samples = 1000
      testX = np.zeros((n_samples, patch_size_d, patch_size_d, 1))
      testlabel = np.zeros((testX.shape[0], 2))
      T, M, N, C = testvideodata.shape
      i = 0
      while i < n_samples:
        print i
        rframe = np.random.randint(0, T-1)
        frames = testvideodata[rframe:rframe+2, :, :, 0].astype(np.float32)
        if (i < n_samples*0.50):
          testlabel[i, 0] = 1
          frame = frames[0].copy()
        else:
          frame = frames[0].copy()
          frame[::2] = frames[1, ::2]
          testlabel[i, 1] = 1

        y = np.random.randint(0, M - patch_size_d)
        x = np.random.randint(0, N - patch_size_d)
        patch = frame[y:y+patch_size_d, x:x+patch_size_d]
        if testlabel[i, 0] == 0:
          mse = np.mean(np.abs(patch[20:-20, 20:-20] - frames[0, y+20:y+patch_size_d-20, x+20:x+patch_size_d-20]))
          if mse < 1:
            continue
        testX[i, :, :, 0] = patch
        i+=1

      for i in range(n_samples):
        testX[i, :, :, 0], _, _ =  skvideo.utils.compute_image_mscn_transform(testX[i, :, :, 0])

      testX = testX[:, 20:-20, 20:-20]
      testX = testX.reshape(-1, 625)
      testX *= kern

      testX = testX.astype(np.float32)

      return saveFolder, trainlabel, trainX, testlabel, testX 

if __name__ == "__main__":
    patch_size = 25

    # sparsity constraint
    lambda_val = np.float(sys.argv[1])

    # labels constraint
    alpha = np.float(sys.argv[2])

    # number of basis functions desired
    n_filters = np.int(sys.argv[3])

    # number of basis functions desired
    experimentName = sys.argv[4]

    # create a dummy database of four crosses elements with noise
    saveFolder, labels, trainX, testlabels, testX = prepData(experimentName)

    n_labels = labels.shape[1]

    dict_path = saveFolder + "/D_" + str(n_filters) + "_" + str(lambda_val) + "_" + str(alpha) + ".pkl"
    dict_path2 = saveFolder + "/D_" + str(n_filters) + "_" + str(lambda_val) + "_" + str(alpha) + ".png"

    np.random.seed(0)

    # initialize weights
    D = np.random.normal(size=(n_filters, patch_size**2))
    for i in range(D.shape[0]):
      D[i, :] /= np.sqrt(np.dot(D[i,:], D[i,:]))
    D = D.astype(np.float32)

    W = np.zeros((n_filters, n_labels), dtype=np.float32)
    b = np.zeros((n_labels, 1), dtype=np.float32)

    # load the latest data for resuming
    start_k = 0
    dual1 = np.zeros((n_filters, trainX.shape[0]), dtype=np.float32)
    primal1 = np.abs(np.random.normal(size=np.shape(dual1))).astype(np.float32)

    ext = [primal1, dual1, D]

    params = [W, b]

    lassoADMMDictCodeSolver(trainX, labels, params, ext, lambda_val, alpha, True, 200)
    joblib.dump(ext[2].reshape(n_filters, 1, patch_size, patch_size), dict_path, compress=9)
    img = plot_weights(ext[2].reshape(n_filters, 1, patch_size, patch_size))
    skimage.io.imsave(dict_path2, img)

    # evaluate performance using an SVM
    trainXprojection = np.dot(trainX, ext[2].T)
    trainXprojection[trainXprojection < 0] = 0
    testXprojection = np.dot(testX, ext[2].T)
    testXprojection[testXprojection < 0] = 0

    # feature normalization...
    mu = np.mean(trainXprojection, axis=0)
    sd = np.std(trainXprojection, axis=0)
    trainXprojection -= mu
    trainXprojection /= 2*sd + 1e-6
    testXprojection -= mu
    testXprojection /= 2*sd + 1e-6

    trainlabels = np.argmax(labels, axis=1) 
    testlabels = np.argmax(testlabels, axis=1) 

    model = sklearn.svm.SVC()
    model.fit(trainXprojection, trainlabels)

    predictedlabels = model.predict(testXprojection)

    print "Accuracy: ", np.mean(predictedlabels == testlabels)
    print "F1 Macro: ", sklearn.metrics.f1_score(testlabels, predictedlabels, average='macro')
