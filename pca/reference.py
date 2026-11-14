"""Python numerical reference for validation only, not the production solver."""
import numpy as np


def power_signature(matrix, initial, iterations):
    """FP64 iteration on exactly the exported FP32 R; FP32 output."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.any(matrix):
        raise ValueError("R must be finite and nonzero")

    def unit(vector):
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("zero or nonfinite power-iteration norm")
        return vector / norm

    right = unit(initial.copy())
    for _ in range(iterations):
        left = unit(matrix @ right)
        right = unit(matrix.T @ left)
    signature = unit(matrix @ right)
    if signature[np.argmax(np.abs(signature))] < 0:
        signature *= -1
    return signature.astype(np.float32)


def lanczos_signature(matrix,initial,steps):
    """Independent NumPy Golub--Kahan reference at the same finite budget."""
    R=np.asarray(matrix,dtype=float)
    if not np.isfinite(R).all() or not np.any(R):raise ValueError('R must be finite and nonzero')
    v=np.asarray(initial,dtype=float).copy();length=np.linalg.norm(v)
    if not length>0 or not np.isfinite(length):raise ValueError('invalid initial vector')
    v/=length;p=R@v;a=np.linalg.norm(p)
    if not a>0:raise ValueError('zero initial left direction')
    U=[p/a];V=[v];alpha=[a];beta=[]
    def reorth(x,basis):
        Q=np.asarray(basis)
        for _ in range(2):x=x-Q.T@(Q@x)
        return x
    for _ in range(steps-1):
        q=reorth(R.T@U[-1]-alpha[-1]*V[-1],V);b=np.linalg.norm(q)
        if b<=1e-13*a:break
        v=q/b;p=reorth(R@v-b*U[-1],U);a=np.linalg.norm(p);beta.append(b)
        if a<=1e-13*b:break
        U.append(p/a);V.append(v);alpha.append(a)
    k=len(U);B=np.zeros((k,k+(len(beta)==k)))
    for i in range(k):
        B[i,i]=alpha[i]
        if i<len(beta):B[i,i+1]=beta[i]
    left,_,_=np.linalg.svd(B,full_matrices=False)
    s=np.asarray(U).T@left[:,0];s/=np.linalg.norm(s)
    if s[np.argmax(np.abs(s))]<0:s=-s
    return s.astype(np.float32)
